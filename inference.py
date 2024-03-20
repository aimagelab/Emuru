import argparse
import logging
import einops
import wandb
from pathlib import Path
import math
import os
from tqdm.auto import tqdm

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision
from torchvision import transforms
from diffusers import AutoencoderKL
from transformers import Trainer
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

import diffusers
from diffusers import AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import is_wandb_available

from PIL import Image

from datasets import OnlineFontSquare, TextSampler

# from models.vae import VAEModel
# from models.configuration_vae import VAEConfig

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__)


def main():
    accelerator = Accelerator()

    model = AutoencoderKL(latent_channels=1, out_channels=1)
    accelerator.load_state('/home/fquattrini/emuru/results/checkpoints/checkpoint_1')

    z = torch.randn(1, 1, 64, 64)
    pred = model.decoder(z)


if __name__ == '__main__':
    main()
