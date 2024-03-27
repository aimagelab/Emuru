import argparse
import logging
import einops
import wandb
from pathlib import Path
import math
import os
from tqdm.auto import tqdm
import uuid
import json

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision

from transformers import Trainer
import accelerate
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

import diffusers
from models.htr import HTR
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import is_wandb_available

from PIL import Image

from utils import TrainState
from custom_datasets import OnlineFontSquare, TextSampler, collate_fn
from models.smooth_ce import SmoothCrossEntropyLoss
import evaluate

# from models.vae import VAEModel
# from models.configuration_vae import VAEConfig

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__)


@torch.no_grad()
def log_validation(eval_loader, htr, accelerator, weight_dtype, loss_fn, cer_fn):
    htr_model = accelerator.unwrap_model(htr)
    htr_model.eval()
    eval_loss = 0.
    cer_value = 0.
    images_for_log = []

    # TODO FONT SQUARE INITIALIZED WITH THE SEED PRODUCES THE SAME TEXT AND IMAGES
    for step, batch in enumerate(eval_loader):
        images = batch['images_bw'].to(weight_dtype)
        text_logits_s2s = batch['text_logits_s2s']  # TODO IMPLEMENT NOISY TEACHER
        tgt_mask = batch['tgt_key_mask']
        tgt_key_padding_mask = batch['tgt_key_padding_mask']

        output = htr(images, text_logits_s2s[:, :-1], tgt_mask, tgt_key_padding_mask[:, :-1])
        loss = loss_fn(output, text_logits_s2s[:, 1:])

        predicted_logits = torch.argmax(output, dim=2)
        eos = torch.tensor(2)  # TODO CHANGE THIS AFTER ALPHABET REFACTORING
        predicted_characters = eval_loader.dataset.alphabet.decode(predicted_logits, [eos])
        correct_characters = eval_loader.dataset.alphabet.decode(text_logits_s2s[:, 1:], [eos])
        cer_value += cer_fn.compute(predictions=predicted_characters, references=correct_characters)
        eval_loss += loss.item()

        if step < 4:
            images_for_log.append(wandb.Image(images[0], caption=predicted_characters[0]))

    accelerator.log({
        "eval_loss": eval_loss / len(eval_loader),
        "cer_value": cer_value / len(eval_loader),
        "images": images_for_log,
    })

    del htr
    torch.cuda.empty_cache()


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='results', help="output directory")
    parser.add_argument("--logging_dir", type=str, default='results', help="logging directory")
    parser.add_argument("--train_batch_size", type=int, default=128, help="train batch size")
    parser.add_argument("--eval_batch_size", type=int, default=128, help="eval batch size")
    parser.add_argument("--epochs", type=int, default=10000, help="number of epochs to train the model")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--seed", type=int, default=24, help="random seed")
    parser.add_argument('--model_save_interval', type=int, default=5, help="model save interval")
    parser.add_argument("--eval_epochs", type=int, default=5, help="eval interval")
    parser.add_argument("--resume_id", type=str, default=None, help="resume from checkpoint")
    parser.add_argument("--htr_config", type=str, default='configs/htr/HTR_64x768.json', help='config path')

    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial",
                                 "constant", "constant_with_warmup"])

    args = parser.parse_args()

    args.mixed_precision = 'no'
    args.use_8bit_adam = False  # todo implement it
    args.gradient_accumulation_steps = 1
    args.checkpoints_total_limit = 5
    args.report_to = "wandb"
    args.wandb_project_name = "emuru_htr"
    args.use_ema = False  # TODO IMPLEMENT IT
    args.scale_lr = False
    args.adam_beta1 = 0.9
    args.adam_beta2 = 0.999
    args.adam_epsilon = 1e-8
    args.adam_weight_decay = 1e-2
    args.lr_warmup_steps = 32
    args.kl_scale = 1e-6
    args.max_grad_norm = 1.0
    args.height = 64
    args.num_samples_per_epoch = None

    args.run_name = args.resume_id if args.resume_id else uuid.uuid4().hex[:4]
    args.output_dir = Path(args.output_dir) / args.run_name
    args.logging_dir = Path(args.logging_dir) / args.run_name

    accelerator_project_config = ProjectConfiguration(
        project_dir=str(args.output_dir),
        logging_dir=str(args.logging_dir),
        automatic_checkpoint_naming=True,
        total_limit=args.checkpoints_total_limit,
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

    with open(args.htr_config, "r") as f:
        config_dict = json.load(f)

    htr = HTR.from_config(config_dict)
    htr.requires_grad_(True)

    if args.scale_lr:
        args.lr = args.lr * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes

    optimizer = torch.optim.Adam(
        htr.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon)

    train_dataset = OnlineFontSquare('files/font_square/fonts', 'files/font_square/backgrounds',
                                     TextSampler(8, 32, 6), length=args.num_samples_per_epoch)
    eval_dataset = OnlineFontSquare('files/font_square/fonts', 'files/font_square/backgrounds',
                                    TextSampler(8, 32, 6), length=1024)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, persistent_workers=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=4, persistent_workers=True)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.epochs * args.gradient_accumulation_steps,
    )

    htr, optimizer, train_loader, eval_loader, lr_scheduler = accelerator.prepare(
        htr, optimizer, train_loader, eval_loader, lr_scheduler)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if accelerator.is_main_process:
        wandb_args = {"wandb": {"entity": "fomo_aiisdh", "name": args.run_name}}
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.wandb_project_name, tracker_config, wandb_args)

    num_update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.epochs * num_update_steps_per_epoch
    total_batch_size = (args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps)
    args.total_params = sum([p.numel() for p in htr.parameters()])

    logger.info("***** Running HTR training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num eval samples = {len(eval_dataset)}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total parameters count = {args.total_params}")

    train_state = TrainState(global_step=0, epoch=0)
    accelerator.register_for_checkpointing(train_state)
    if args.resume_id:
        accelerator.load_state()
        accelerator.project_configuration.iteration = train_state.epoch

    progress_bar = tqdm(range(train_state.global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process)  # TODO SHOULD HAVE ONE PROGRESS BAR FOR EACH EPOCH
    progress_bar.set_description("Steps") # TODO IN WANDB I SHOULD HAVE SECTIONS

    wandb.watch(htr, log="all", log_freq=1)
    smooth_ce_loss = SmoothCrossEntropyLoss(tgt_pad_idx=0)
    cer = evaluate.load('cer')

    for epoch in range(train_state.epoch, args.epochs):
        htr.train()
        train_loss = 0.
        train_cer = 0.

        for step, batch in enumerate(train_loader):

            with accelerator.accumulate(htr):
                images = batch['images_bw'].to(weight_dtype)
                text_logits_s2s = batch['text_logits_s2s']  # TODO IMPLEMENT NOISY TEACHER
                tgt_mask = batch['tgt_key_mask']
                tgt_key_padding_mask = batch['tgt_key_padding_mask']

                # x, tgt_logits, tgt_mask, tgt_key_padding_mask  # TODO CANCEL THIS
                output = htr(images, text_logits_s2s[:, :-1], tgt_mask, tgt_key_padding_mask[:, :-1])

                loss = smooth_ce_loss(output, text_logits_s2s[:, 1:])

                predicted_logits = torch.argmax(output, dim=2)
                eos = torch.tensor(2)  # TODO CHANGE THIS AFTER ALPHABET REFACTORING
                predicted_characters = train_dataset.alphabet.decode(predicted_logits, [eos])
                correct_characters = train_dataset.alphabet.decode(text_logits_s2s[:, 1:], [eos])
                cer_value = cer.compute(predictions=predicted_characters, references=correct_characters)

                if not torch.isfinite(loss):
                    logger.info("\nWARNING: non-finite loss")
                    optimizer.zero_grad()
                    continue

                avg_loss = accelerator.gather(loss).mean()
                avg_cer = accelerator.gather(torch.tensor(cer_value)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                train_cer += avg_cer.item() / args.gradient_accumulation_steps
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(htr.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            logs = {}
            if accelerator.sync_gradients:
                progress_bar.update(1)
                train_state.global_step += 1
                logs["global_step"] = train_state.global_step
                logs['train_loss'] = train_loss
                logs['train_cer'] = train_cer
                train_loss = 0.0
                train_cer = 0.0

            logs["lr"] = lr_scheduler.get_last_lr()[0]
            logs["smooth_ce"] = loss.detach().item()
            logs["cer_value"] = cer_value
            logs['epoch'] = epoch

            accelerator.log(logs)
            progress_bar.set_postfix(**logs)

        train_state.epoch += 1
        if accelerator.is_main_process:
            if epoch % args.eval_epochs == 0:
                accelerator.save_state()
                with torch.no_grad():
                    log_validation(eval_loader, htr, accelerator, weight_dtype, loss_fn=smooth_ce_loss, cer_fn=cer)

            if epoch % args.model_save_interval == 0:
                vae = accelerator.unwrap_model(htr)
                vae.save_pretrained(args.output_dir / f"model_{epoch:04d}")

    if accelerator.is_main_process:
        vae = accelerator.unwrap_model(htr)
        vae.save_pretrained(args.output_dir)

    accelerator.end_training()
    logger.info("***** Training finished *****")


if __name__ == "__main__":
    train()
