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
import json
from PIL import Image
from torchvision.utils import save_image

# from models.vae import VAEModel
# from models.configuration_vae import VAEConfig

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__)


@torch.inference_mode()
def main():
    vae = AutoencoderKL.from_pretrained('files/checkpoints/ca5f')

    img = Image.open('test.png')
    img = transforms.ToTensor()(img).unsqueeze(0)
    img = img * 2 - 1
    posterior = vae.encode(img).latent_dist
    z = posterior.sample()

    # z = torch.randn(1, 1, 8, 64)
    pred = vae.decoder(z)
    save_image(pred[0], 'test_recon.png')
    save_image(pred[0] * -1, 'test_recon_2.png')
    print()


if __name__ == '__main__':
    main()
