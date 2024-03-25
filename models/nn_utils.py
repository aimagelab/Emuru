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


class PositionalEncoding1D(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout=0.1, max_len=1000, positional_scaler =1.0):
        super(PositionalEncoding1D, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model, device='cuda')
        position = torch.arange(0, max_len,device='cuda').unsqueeze(1) * positional_scaler
        div_term = torch.exp(torch.arange(0, d_model, 2,device='cuda') *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        shape of x has to be: (N, S, F)   # N = batch size, S = sequence length, F = feature dimension
        """
        pe = Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        x = x + pe
        return self.dropout(x)