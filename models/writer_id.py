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

from .vae import Encoder


# Should have the following structure:
#  | Name                   | Type               | Params
# -------------------------------------------------
# 0 | feature_extractor     | Encoder            | 2.9 M
# 1 | linear                | Linear             | 65.8 K
# 2 | relu                  | LeakyReLU          | 0
# 3 | linear2               | Linear             | 29.8 M
# -------------------------------------------------
# This implementation has 32.767.273 parameters with 115961 fonts
class WriterID(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(self,
                 num_writers: int = 10400,
                 in_channels: int = 1,
                 down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
                 block_out_channels: Tuple[int] = (64,),
                 layers_per_block: int = 1,
                 act_fn: str = "silu",
                 latent_channels: int = 256,
                 norm_num_groups: int = 16,
                 encoder_dropout: float = 0.1,
                 only_head: bool = False,
                 ):
        super(WriterID, self).__init__()

        self.only_head = only_head
        if not self.only_head:
            self.feature_extractor = Encoder(
                in_channels=in_channels,
                out_channels=latent_channels,
                down_block_types=down_block_types,
                block_out_channels=block_out_channels,
                layers_per_block=layers_per_block,
                act_fn=act_fn,
                norm_num_groups=norm_num_groups,
                dropout=encoder_dropout,
                mid_block_add_attention=False,
                double_z=False,
                add_mid_block=False)
        else:
            self.conv_out = nn.Conv2d(in_channels, latent_channels, 3, padding=1)

        self.linear = nn.Linear(latent_channels, latent_channels)
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(latent_channels, num_writers)

    def forward(self, x):
        if not self.only_head:
            x = self.feature_extractor(x)  # [B, 1, 64, 768]  -->  [B, 256, 1, 12]
        else:
            x = self.conv_out(x)

        # if latent writer_id then input is [B, 1, 8, 96]
        out = torch.mean(x, dim=[-1, -2])
        out = self.linear2(self.relu(self.linear(out)))

        return out
