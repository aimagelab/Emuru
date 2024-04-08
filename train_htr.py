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
from diffusers.training_utils import EMAModel

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.utils import broadcast

from transformers.optimization import get_scheduler
import evaluate

from utils import TrainState
from custom_datasets import OnlineFontSquare, TextSampler, collate_fn
from models.smooth_ce import SmoothCrossEntropyLoss
from custom_datasets.constants import END_OF_SEQUENCE
from models.htr import HTR
from models.teacher_forcing import NoisyTeacherForcing

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__)


@torch.no_grad()
def validation(eval_loader, htr, accelerator, weight_dtype, loss_fn, cer_fn, wandb_prefix="eval"):
    htr_model = accelerator.unwrap_model(htr)
    htr_model.eval()
    eval_loss = 0.
    images_for_log = []

    for step, batch in enumerate(eval_loader):
        with accelerator.autocast():
            images = batch['images_bw'].to(weight_dtype)
            text_logits_s2s = batch['text_logits_s2s']
            tgt_mask = batch['tgt_key_mask']
            tgt_key_padding_mask = batch['tgt_key_padding_mask']

            output = htr_model(images, text_logits_s2s[:, :-1], tgt_mask, tgt_key_padding_mask[:, :-1])
            loss = loss_fn(output, text_logits_s2s[:, 1:])

            predicted_logits = torch.argmax(output, dim=2)
            predicted_characters = eval_loader.dataset.alphabet.decode(predicted_logits, [END_OF_SEQUENCE])
            correct_characters = eval_loader.dataset.alphabet.decode(text_logits_s2s[:, 1:], [END_OF_SEQUENCE])

            cer_fn.add_batch(predictions=predicted_characters, references=correct_characters)
            eval_loss += loss.item()

            if step < 4:
                images_for_log.append(wandb.Image(images[0], caption=predicted_characters[0]))

    cer_value = cer_fn.compute()
    accelerator.log({
        f"{wandb_prefix}/loss": eval_loss / len(eval_loader),
        f"{wandb_prefix}/cer": cer_value / len(eval_loader),
        f"{wandb_prefix}/images": images_for_log,
    })

    del htr_model
    torch.cuda.empty_cache()
    return cer_value


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='results', help="output directory")
    parser.add_argument("--logging_dir", type=str, default='results', help="logging directory")
    parser.add_argument("--train_batch_size", type=int, default=128, help="train batch size")
    parser.add_argument("--eval_batch_size", type=int, default=128, help="eval batch size")
    parser.add_argument("--epochs", type=int, default=10000, help="number of train epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--seed", type=int, default=24, help="random seed")
    parser.add_argument('--model_save_interval', type=int, default=5, help="model save interval")
    parser.add_argument('--wandb_log_interval_steps', type=int, default=25, help="model save interval")
    parser.add_argument("--eval_epochs", type=int, default=5, help="eval interval")
    parser.add_argument("--resume_id", type=str, default=None, help="resume from checkpoint")
    parser.add_argument("--htr_config", type=str, default='configs/htr/HTR_64x768.json', help='config path')
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--wandb_project_name", type=str, default="emuru_htr", help="wandb project name")

    parser.add_argument("--num_samples_per_epoch", type=int, default=None)
    parser.add_argument("--lr_scheduler", type=str, default="reduce_lr_on_plateau")
    parser.add_argument("--lr_scheduler_patience", type=int, default=10)
    parser.add_argument("--use_ema", type=str, default="True")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="no")
    parser.add_argument("--checkpoints_total_limit", type=int, default=5)

    args = parser.parse_args()

    args.use_ema = args.use_ema == "True"
    args.adam_beta1 = 0.9
    args.adam_beta2 = 0.999
    args.adam_epsilon = 1e-8
    args.adam_weight_decay = 0

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
        cpu=False,
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

    if args.use_ema:
        ema_htr = HTR.from_config(config_dict)
        ema_htr = EMAModel(ema_htr.parameters(), model_cls=HTR, model_config=htr.config)
        accelerator.register_for_checkpointing(ema_htr)

    optimizer = torch.optim.AdamW(
        htr.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon)

    train_dataset = OnlineFontSquare('files/font_square/fonts', 'files/font_square/backgrounds',
                                     text_sampler=TextSampler(8, 32, (4, 7), exponent=0.5),
                                     length=args.num_samples_per_epoch)
    eval_dataset = OnlineFontSquare('files/font_square/fonts', 'files/font_square/backgrounds',
                                    text_sampler=TextSampler(8, 32, (4, 7), exponent=0.5),
                                    length=args.num_samples_per_epoch)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, persistent_workers=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=4, persistent_workers=True)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        scheduler_specific_kwargs={"patience": args.lr_scheduler_patience}
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
        wandb.watch(htr, log="all", log_freq=1)

    num_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.epochs * num_steps_per_epoch
    total_batch_size = (args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps)
    args.total_params = sum([p.numel() for p in htr.parameters()])

    logger.info("***** Running HTR training *****")
    logger.info(f"  Num train samples = {len(train_dataset)}. Num steps per epoch = {num_steps_per_epoch}")
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

    smooth_ce_loss = SmoothCrossEntropyLoss(tgt_pad_idx=train_dataset.alphabet.pad)
    cer = evaluate.load('cer')
    noisy_teacher = NoisyTeacherForcing(len(train_dataset.alphabet), train_dataset.alphabet.num_extra_tokens, 0.1)

    progress_bar = tqdm(range(train_state.global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(train_state.epoch, args.epochs):

        htr.train()
        train_loss = 0.
        train_cer = 0.

        for step, batch in enumerate(train_loader):

            with accelerator.autocast():
                with accelerator.accumulate(htr):
                    images = batch['images_bw'].to(weight_dtype)
                    text_logits_s2s = batch['text_logits_s2s']
                    text_logits_s2s_noisy = noisy_teacher(text_logits_s2s, batch['unpadded_texts_len'])
                    tgt_mask = batch['tgt_key_mask']
                    tgt_key_padding_mask = batch['tgt_key_padding_mask']

                    output = htr(images, text_logits_s2s_noisy[:, :-1], tgt_mask, tgt_key_padding_mask[:, :-1])

                    loss = smooth_ce_loss(output, text_logits_s2s[:, 1:])

                    predicted_logits = torch.argmax(output, dim=2)
                    predicted_characters = train_dataset.alphabet.decode(predicted_logits, [END_OF_SEQUENCE])
                    correct_characters = train_dataset.alphabet.decode(text_logits_s2s[:, 1:], [END_OF_SEQUENCE])
                    cer_value = cer.compute(predictions=predicted_characters, references=correct_characters)

                    if not torch.isfinite(loss):
                        logger.info("\nWARNING: non-finite loss")
                        optimizer.zero_grad()
                        continue

                    avg_loss = accelerator.gather(loss).mean()
                    avg_cer = accelerator.gather(torch.tensor(cer_value, device=accelerator.device)).mean()
                    train_loss += avg_loss.item() / args.gradient_accumulation_steps
                    train_cer += avg_cer.item() / args.gradient_accumulation_steps
                    accelerator.backward(loss)

                    optimizer.step()
                    optimizer.zero_grad()

                logs = {}
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    if args.use_ema:
                        ema_htr.to(htr.device)
                        ema_htr.step(htr.parameters())
                    train_state.global_step += 1
                    logs["global_step"] = train_state.global_step
                    logs['train/loss'] = train_loss
                    logs['train/cer'] = train_cer
                    train_loss = 0.0
                    train_cer = 0.0

                logs["lr"] = optimizer.param_groups[0]['lr']
                logs["train/smooth_ce"] = loss.detach().item()
                logs["train/cer"] = cer_value
                logs['epoch'] = epoch

                progress_bar.set_postfix(**logs)
                if train_state.global_step % args.wandb_log_interval_steps == 0:
                    accelerator.log(logs)

        train_state.epoch += 1

        print(f'{accelerator.device} before eval')
        if epoch % args.eval_epochs == 0 and accelerator.is_main_process:
            with torch.no_grad():
                eval_cer = validation(eval_loader, htr, accelerator, weight_dtype, smooth_ce_loss, cer, 'eval')
                eval_cer = broadcast(torch.tensor(eval_cer, device=accelerator.device), from_process=0)

                if args.use_ema:
                    ema_htr.store(htr.parameters())
                    ema_htr.copy_to(htr.parameters())
                    _ = validation(eval_loader, htr, accelerator, weight_dtype, smooth_ce_loss, cer, 'ema')
                    ema_htr.restore(htr.parameters())

            logger.info(f"Epoch {epoch} - Eval CER: {eval_cer}")
            accelerator.save_state()

        print(f'{accelerator.device} waiting for everyone')
        accelerator.wait_for_everyone()
        print(f'{accelerator.device} continuing')
        if accelerator.is_main_process and epoch % args.model_save_interval == 0:
            htr_to_save = accelerator.unwrap_model(htr)
            htr_to_save.save_pretrained(args.output_dir / f"model_{epoch:04d}")
            del htr_to_save

        lr_scheduler.step(eval_cer)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        htr = accelerator.unwrap_model(htr)
        htr.save_pretrained(args.output_dir)

        if args.use_ema:
            ema_htr.copy_to(htr.parameters())
            htr.save_pretrained(args.output_dir / f"ema")

    accelerator.end_training()
    logger.info("***** Training finished *****")


if __name__ == "__main__":
    train()
