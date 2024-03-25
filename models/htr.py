from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

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

from vae import Encoder


class HTR(nn.Module):
    def __init__(self,
                 alphabet_size: int = 169,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
                 up_block_types: Tuple[str] = ("UpDecoderBlock2D",),
                 block_out_channels: Tuple[int] = (64,),
                 layers_per_block: int = 1,
                 act_fn: str = "silu",
                 latent_channels: int = 128,
                 d_model: int = 128,
                 norm_num_groups: int = 16,
                 encoder_dropout: float = 0.1,
                 tgt_pe=True,
                 mem_pe=True,
                 htr_dropout: float = 0.1
                 ):
        super(HTR, self).__init__()

        self.feature_extractor = Encoder()

        self.feature_extractor = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=False,
            dropout=encoder_dropout,
        )

        self.quant_conv = nn.Conv2d(latent_channels, d_model, 1)

        # Letter classification
        self.text_embedding = nn.Embedding(alphabet_size, d_model)


