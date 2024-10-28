from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from diffusers.utils import BaseOutput, is_torch_version
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import SpatialNorm
from diffusers.models.unets.unet_2d_blocks import (
    AutoencoderTinyBlock,
    UNetMidBlock2D,
    get_down_block,
    get_up_block,
)

from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from .vae import Encoder
from .nn_utils import PositionalEncoding1D, A2DPE


# Should have the following structure:
#  | Name      | Type               | Params
# -------------------------------------------------
# 0 | model     | Sequential         | 1.9 M
# 1 | linear    | Linear             | 65.8 K
# 2 | relu      | LeakyReLU          | 0
# 3 | linear2   | Linear             | 2.7 M
# 4 | criterion | CrossEntropyLoss   | 0
# 5 | acc       | MulticlassAccuracy | 0
# -------------------------------------------------
# This implementation has 4.786.464 parameters
class WriterID(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(self,
                 num_writers: int = 10400,
                 in_channels: int = 1,
                 block_out_channels: Tuple[int] = (64,),
                 latent_channels: int = 256,
                 norm_num_groups: int = 16,
                 encoder_dropout: float = 0.1,
                 ):
        super(WriterID, self).__init__()

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.down_blocks = nn.ModuleList([])
        output_channel = block_out_channels[0]
        num_blocks = len(block_out_channels) - 1
        for i in range(1, num_blocks):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == num_blocks - 1
            is_second_to_last_block = i == num_blocks - 2

            self.down_blocks.append(
                ResnetBlock(in_channels=input_channel,
                            out_channels=output_channel,
                            temb_channels=0,
                            dropout=encoder_dropout,
                            group_norm=norm_num_groups))

            if is_final_block:
                self.down_blocks.append(Downsample(output_channel, True, down_sample_factor=(1, 2)))
            elif is_second_to_last_block:
                self.down_blocks.append(Downsample(output_channel, True, down_sample_factor=(2, 4)))
            else:
                self.down_blocks.append(Downsample(output_channel, True, down_sample_factor=(4, 4)))

        self.down_blocks.append(
            ResnetBlock(in_channels=block_out_channels[-1],
                        out_channels=latent_channels,
                        temb_channels=0,
                        dropout=encoder_dropout,
                        group_norm=norm_num_groups))

        self.down_blocks = nn.Sequential(*self.down_blocks)

        self.linear = nn.Linear(latent_channels, latent_channels)
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(latent_channels, num_writers)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.down_blocks(x)  # [B, 1, 64, 768]  -->  [B, 256, 1, 6]

        out = torch.mean(x, dim=[-1, -2])
        out = self.linear2(self.relu(self.linear(out)))

        return out
    
    def reset_last_layer(self, num_writers: int):
        self.linear2 = nn.Linear(self.config.latent_channels, num_writers)


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512, group_norm=16):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = get_group_norm(num_groups=group_norm, num_channels=in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = get_group_norm(num_groups=group_norm, num_channels=out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)

        self.non_linearity = nn.SiLU()
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb=None):
        h = x
        h = self.norm1(h)
        h = self.non_linearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(self.non_linearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = self.non_linearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class Downsample(nn.Module):
    def __init__(self, in_channels: int, with_conv: bool, down_sample_factor: Tuple[int, int] = (2, 2)):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=down_sample_factor,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


def get_group_norm(num_channels, num_groups=32):
    if num_channels < num_groups:
        num_groups = num_channels
    return torch.nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, eps=1e-5, affine=True)
