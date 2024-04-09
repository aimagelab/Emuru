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
from torchvision import transforms
from custom_datasets import OnlineFontSquare, TextSampler, collate_fn

import argparse
import logging
import wandb
from pathlib import Path
import math
from tqdm.auto import tqdm
import uuid
import json

import torch
from torch.utils.data import DataLoader
import torch.utils.checkpoint
import torchvision

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.utils import broadcast

from models.autoencoder_kl import AutoencoderKL
from transformers.optimization import get_scheduler
from diffusers.training_utils import EMAModel

from utils import TrainState
from custom_datasets import OnlineFontSquare, TextSampler, collate_fn
from models.autoencoder_loss import AutoencoderLoss

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
    vae = AutoencoderKL.from_pretrained('/home/fquattrini/emuru/results/228a/model_0500')

    img = Image.open(r'/home/fquattrini/Teddy/files/iam/168/test_168_0000.png').convert('RGB')
    img = img.resize((img.size[0] * 64 // img.size[1], 64))
    img = transforms.ToTensor()(img).unsqueeze(0)
    images = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)

    posterior = vae.encode(images).latent_dist
    z = posterior.sample()
    # z = torch.randn(1, 1, 8, 64)
    pred = vae.decode(z).sample
    print()


if __name__ == '__main__':
    main()
