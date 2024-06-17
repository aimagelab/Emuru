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


# Parameters should be 5.438.209
# 0 | feature_extractor | Encoder       | 1.6 M
# 1 | quant_conv        | Conv2d        | 16.5 K
# 2 | htr               | HTR           | 3.9 M

# Parameters: 5.437.695
class HTR(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(self,
                 alphabet_size: int = 169,
                 in_channels: int = 3,
                 down_block_types: Tuple[str] = ("DownEncoderBlock2D",),
                 block_out_channels: Tuple[int] = (64,),
                 layers_per_block: int = 1,
                 act_fn: str = "silu",
                 latent_channels: int = 128,
                 d_model: int = 128,
                 norm_num_groups: int = 16,
                 encoder_dropout: float = 0.1,
                 use_tgt_pe=True,
                 use_mem_pe=True,
                 htr_dropout: float = 0.1,
                 num_encoder_layers: int = 2,
                 num_decoder_layers: int = 4,
                 
                 ):
        super(HTR, self).__init__()

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
        self.d_model = d_model
        self.mem_pe = A2DPE(d_model=d_model, dropout=htr_dropout) if use_mem_pe else None
        self.tgt_pe = PositionalEncoding1D(d_model=d_model, dropout=htr_dropout) if use_tgt_pe else None

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=1)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=1)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_decoder_layers, norm=nn.LayerNorm(d_model))

        self.fc = nn.Linear(d_model, alphabet_size)

    def forward(self, x, tgt_logits, tgt_mask, tgt_key_padding_mask):
        # Feature extraction
        memory = self.feature_extractor(x)  # [16, 1, 64, 768] -> [16, 128, 8, 96]
        memory = self.quant_conv(memory)

        # Letter classification
        if self.mem_pe is not None:
            memory = self.mem_pe(memory)

        memory = rearrange(memory, "b c h w -> (h w) b c")
        memory = self.transformer_encoder(memory)

        tgt = self.text_embedding(tgt_logits)
        if self.tgt_pe is not None:
            tgt = self.tgt_pe(tgt)

        tgt = rearrange(tgt, "b s d -> s b d")
        tgt = self.transformer_decoder(tgt, memory, tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask.float())
        tgt = rearrange(tgt, "s b d -> b s d")
        tgt = self.fc(tgt)

        return tgt
