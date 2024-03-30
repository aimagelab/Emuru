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
                 down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
                 block_out_channels: Tuple[int] = (64,),
                 layers_per_block: int = 1,
                 act_fn: str = "silu",
                 latent_channels: int = 256,
                 norm_num_groups: int = 16,
                 encoder_dropout: float = 0.1,
                 ):
        super(WriterID, self).__init__()

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
            add_mid_block=False
        )

        # sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.linear = nn.Linear(latent_channels, latent_channels)
        self.relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(latent_channels, num_writers)

    def forward(self, x):
        x = self.feature_extractor(x)

        out = torch.mean(x, dim=[-1, -2])
        out = self.linear2(self.relu(self.linear(out)))

        return out
