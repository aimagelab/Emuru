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


@torch.no_grad()
def log_validation(args, eval_loader, vae, accelerator, weight_dtype, epoch):
    vae_model = accelerator.unwrap_model(vae)
    vae_model.eval()
    eval_loss = 0.
    images = []

    for step, batch in enumerate(eval_loader):
        image = batch['img'].to(weight_dtype)
        target = batch['bw_img'].to(weight_dtype)

        if accelerator.num_processes > 1:
            posterior = vae_model.module.encode(image).latent_dist
        else:
            posterior = vae_model.encode(image).latent_dist

        z = posterior.sample()

        pred = vae_model.module.decode(z).sample if accelerator.num_processes > 1 else vae_model.decode(z).sample
        kl_loss = posterior.kl().mean()
        mse_loss = F.mse_loss(pred, target[:, :, :, :pred.shape[3]], reduction='mean')

        loss = mse_loss + args.kl_scale * kl_loss
        eval_loss += loss.item()

        if step == 0:
            images.append(torch.cat([image.cpu(), pred.repeat(1, 3, 1, 1).cpu()], dim=-1)[:8])

    grid_nrows = 2
    accelerator.log({
        "eval_loss": eval_loss / len(eval_loader),
        "Original (left), Reconstruction (right)":
            [wandb.Image(torchvision.utils.make_grid(image, nrow=grid_nrows, normalize=True, value_range=(-1, 1))) for _, image in enumerate(images)]
    })

    del vae_model
    torch.cuda.empty_cache()


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='results', help="output directory")
    parser.add_argument("--logging_dir", type=str, default='results', help="logging directory")
    parser.add_argument("--train_batch_size", type=int, default=16, help="train batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="eval batch size")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs to train the model")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--seed", type=int, default=24, help="random seed")
    parser.add_argument("--log_interval", type=int, default=100, help="log interval")
    parser.add_argument("--eval_epochs", type=int, default=10, help="eval interval")

    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial",
                                 "constant", "constant_with_warmup"])

    args = parser.parse_args()

    args.mixed_precision = 'bf16'
    args.resolution = 64
    args.use_8bit_adam = False  # todo implement it
    args.gradient_accumulation_steps = 1
    args.checkpoints_total_limit = 5
    args.report_to = "wandb"
    args.wandb_project_name = "vae_htg"
    args.use_ema = False  # TODO IMPLEMENT IT
    args.gradient_checkpointing = False
    args.scale_lr = False
    args.adam_beta1 = 0.9
    args.adam_beta2 = 0.999
    args.adam_epsilon = 1e-8
    args.adam_weight_decay = 1e-2
    args.lr_warmup_steps = 32
    args.kl_scale = 1e-6
    args.max_grad_norm = 1.0
    args.height = 64

    accelerator_project_config = ProjectConfiguration(
        total_limit=args.checkpoints_total_limit,
        project_dir=str(args.output_dir),
        logging_dir=str(args.logging_dir),
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        args.output_dir = Path(args.output_dir)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.logging_dir = Path(args.logging_dir)
        args.logging_dir.mkdir(parents=True, exist_ok=True)

    vae = AutoencoderKL(latent_channels=1, out_channels=1)  # TODO VAE CONFIG
    # vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2", subfolder='vae')
    vae.requires_grad_(True)
    if args.gradient_checkpointing:
        vae.enable_gradient_checkpointing()
    if args.scale_lr:
        args.lr = args.lr * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes

    optimizer = torch.optim.AdamW(
        vae.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon)

    train_dataset = OnlineFontSquare('files/font_square/fonts', 'files/font_square/backgrounds', TextSampler(8, 32, 6))
    eval_dataset = OnlineFontSquare('files/font_square/fonts', 'files/font_square/backgrounds', TextSampler(8, 32, 6), length=64)

    eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=eval_dataset.collate_fn, num_workers=4)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=False, collate_fn=train_dataset.collate_fn, num_workers=4)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.epochs * args.gradient_accumulation_steps,
    )

    vae, vae.encoder, vae.decoder, optimizer, train_loader, eval_loader, lr_scheduler = (
        accelerator.prepare(vae, vae.encoder, vae.decoder, optimizer, train_loader, eval_loader, lr_scheduler))

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.wandb_project_name, tracker_config)

    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.epochs * num_update_steps_per_epoch
    total_batch_size = (args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps)
    args.total_params = sum([p.numel() for p in vae.parameters()])

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num eval samples = {len(eval_dataset)}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total parameters count = {args.total_params}")

    global_step = 0
    starting_epoch = 0

    # todo resume

    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(starting_epoch, args.epochs):
        vae.train()
        train_loss = 0.

        for step, batch in enumerate(train_loader):
            with accelerator.autocast():
                with accelerator.accumulate(vae):
                    image = batch['img'].to(weight_dtype)
                    target = batch['bw_img'].to(weight_dtype)

                    if accelerator.num_processes > 1:
                        posterior = vae.module.encode(image).latent_dist
                    else:
                        posterior = vae.encode(image).latent_dist

                    z = posterior.sample()

                    pred = vae.module.decode(z).sample if accelerator.num_processes > 1 else vae.decode(z).sample
                    kl_loss = posterior.kl().mean()
                    mse_loss = F.mse_loss(pred, target[:, :, :, :pred.shape[3]], reduction='mean')

                    loss = mse_loss + args.kl_scale * kl_loss

                    if not torch.isfinite(loss):
                        pred_mean = pred.mean()
                        target_mean = target.mean()
                        logger.info("\nWARNING: non-finite loss, ending training ")

                    avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                    train_loss += avg_loss.item() / args.gradient_accumulation_steps
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(vae.parameters(), args.max_grad_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    accelerator.log({"train_loss": train_loss})
                    train_loss = 0.0

                logs = {
                    "step_loss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "mse": mse_loss.detach().item(),
                    "kl": kl_loss.detach().item(),
                    'epoch': epoch,
                }

                accelerator.log(logs)
                progress_bar.set_postfix(**logs)

                if accelerator.is_main_process:
                    accelerator.save_model(accelerator.unwrap_model(vae), save_directory='results')
                    if epoch % args.eval_epochs == 0:
                        with torch.no_grad():
                            log_validation(args, eval_loader, vae, accelerator, weight_dtype, epoch)

    accelerator.end_training()
    logger.info("***** Training finished *****")


if __name__ == "__main__":
    train()
