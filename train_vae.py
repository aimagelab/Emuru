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

from models.autoencoder_kl import AutoencoderKL
from transformers.optimization import get_scheduler
from diffusers.training_utils import EMAModel

from utils import TrainState
from custom_datasets import OnlineFontSquare, TextSampler, collate_fn
from models.autoencoder_loss import AutoencoderLoss

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__)


@torch.no_grad()
def validation(eval_loader, vae, accelerator, weight_dtype, loss_fn, global_step, wandb_prefix="eval"):
    vae.eval()
    eval_loss = 0.
    images_for_log = []
    images_for_log_w_htr_wid = []

    for step, batch in enumerate(eval_loader):
        images = batch['images'].to(weight_dtype)
        targets = batch['images_bw'].to(weight_dtype)
        writers = batch['writers']

        text_logits_s2s = batch['text_logits_s2s']
        text_logits_s2s_unpadded_len = batch['unpadded_texts_len']
        tgt_mask = batch['tgt_key_mask']
        tgt_key_padding_mask = batch['tgt_key_padding_mask']

        posterior = vae.encode(images).latent_dist
        z = posterior.sample()
        pred = vae.decode(z).sample

        loss, log_dict, wandb_media_log = loss_fn(images=targets, reconstructions=pred, posteriors=posterior,
                                                  writers=writers, text_logits_s2s=text_logits_s2s,
                                                  text_logits_s2s_length=text_logits_s2s_unpadded_len,
                                                  tgt_key_padding_mask=tgt_key_padding_mask, source_mask=tgt_mask,
                                                  split=wandb_prefix)

        eval_loss += loss.item()

        if step == 0:
            images_for_log.append(torch.cat([images.cpu(), pred.repeat(1, 3, 1, 1).cpu()], dim=-1)[:8])

        if step < 2:
            author_id = batch['writers'][0].item()
            pred_author_id = wandb_media_log[f'{wandb_prefix}/predicted_authors'][0][0]
            text = batch['texts'][0]
            pred_text = wandb_media_log[f'{wandb_prefix}/predicted_characters'][0][0]
            images_for_log_w_htr_wid.append(wandb.Image(
                images.cpu()[0],
                caption=f'AID: {author_id}, Pred AID: {pred_author_id}, Text: {text}, Pred Text: {pred_text}')
            )

    grid_nrows = 2

    if accelerator.is_main_process:
        accelerator.log({
            **log_dict,
            f"{wandb_prefix}/loss": eval_loss / len(eval_loader),
            "Original (left), Reconstruction (right)":
                [wandb.Image(torchvision.utils.make_grid(image, nrow=grid_nrows, normalize=True, value_range=(-1, 1)))
                 for _, image in enumerate(images_for_log)],
            "Image, HTR and Writer ID": images_for_log_w_htr_wid
        })

    torch.cuda.empty_cache()
    return eval_loss / len(eval_loader)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='results', help="output directory")
    parser.add_argument("--logging_dir", type=str, default='results', help="logging directory")
    parser.add_argument("--train_batch_size", type=int, default=1, help="train batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="eval batch size")
    parser.add_argument("--epochs", type=int, default=10000, help="number of epochs to train the model")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--seed", type=int, default=24, help="random seed")
    parser.add_argument('--model_save_interval', type=int, default=5, help="model save interval")
    parser.add_argument("--eval_epochs", type=int, default=5, help="eval interval")
    parser.add_argument("--resume_id", type=str, default=None, help="resume from checkpoint")
    parser.add_argument("--vae_config", type=str, default='configs/vae/VAE_64x768.json', help='vae config path')
    parser.add_argument("--htr_path", type=str, default='results/8da9/model_1000', help='htr checkpoint path')
    parser.add_argument("--writer_id_path", type=str, default='results/b12a/model_2900', help='writerid config path')
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--wandb_project_name", type=str, default="emuru_vae", help="wandb project name")

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
    args.kl_scale = 1e-6

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

    with open(args.vae_config, "r") as f:
        config_dict = json.load(f)

    vae = AutoencoderKL.from_config(config_dict)
    vae.requires_grad_(True)

    if args.use_ema:
        ema_vae = vae.from_config(config_dict)
        ema_vae = EMAModel(ema_vae.parameters(), model_cls=AutoencoderKL, model_config=vae.config)
        accelerator.register_for_checkpointing(ema_vae)

    optimizer = torch.optim.AdamW(
        vae.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon)

    train_dataset = OnlineFontSquare('files/font_square/fonts', 'files/font_square/backgrounds',
                                     TextSampler(8, 32, (4, 7)), length=args.num_samples_per_epoch)
    eval_dataset = OnlineFontSquare('files/font_square/fonts', 'files/font_square/backgrounds',
                                    TextSampler(8, 32, (4, 7)), length=1024)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, persistent_workers=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=4, persistent_workers=True)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        scheduler_specific_kwargs={"patience": args.lr_scheduler_patience}
    )

    loss_fn = AutoencoderLoss(alphabet=train_dataset.alphabet, htr_path=args.htr_path,
                              writer_id_path=args.writer_id_path)

    vae, optimizer, train_loader, eval_loader, lr_scheduler, loss_fn = accelerator.prepare(
        vae, optimizer, train_loader, eval_loader, lr_scheduler, loss_fn)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if accelerator.is_main_process:
        wandb_args = {"wandb": {"entity": "fomo_aiisdh", "name": args.run_name}}
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.wandb_project_name, tracker_config, wandb_args)

    num_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.epochs * num_steps_per_epoch
    total_batch_size = (args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps)
    args.total_params = vae.num_parameters(only_trainable=True)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}. Num steps per epoch = {num_steps_per_epoch}")
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

    wandb.watch(vae, log="all", log_freq=1)

    progress_bar = tqdm(range(train_state.global_step, args.max_train_steps),
                        disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(train_state.epoch, args.epochs):
        vae.train()
        train_loss = 0.

        for step, batch in enumerate(train_loader):

            with accelerator.autocast():
                with accelerator.accumulate(vae):
                    images = batch['images'].to(weight_dtype)
                    targets = batch['images_bw'].to(weight_dtype)
                    writers = batch['writers']

                    text_logits_s2s = batch['text_logits_s2s']
                    text_logits_s2s_unpadded_len = batch['unpadded_texts_len']
                    tgt_mask = batch['tgt_key_mask']
                    tgt_key_padding_mask = batch['tgt_key_padding_mask']

                    posterior = vae.encode(images).latent_dist
                    z = posterior.sample()
                    pred = vae.decode(z).sample

                    loss, log_dict, _ = loss_fn(images=targets, reconstructions=pred, posteriors=posterior,
                                                writers=writers, text_logits_s2s=text_logits_s2s,
                                                text_logits_s2s_length=text_logits_s2s_unpadded_len,
                                                tgt_key_padding_mask=tgt_key_padding_mask, source_mask=tgt_mask,
                                                split="train")

                    if not torch.isfinite(loss):
                        logger.info("\nWARNING: non-finite loss")
                        optimizer.zero_grad()
                        continue

                    avg_loss = accelerator.gather(loss).mean()
                    train_loss += avg_loss.item() / args.gradient_accumulation_steps
                    accelerator.backward(loss)

                    optimizer.step()
                    optimizer.zero_grad()

                logs = {}
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    if args.use_ema:
                        ema_vae.to(vae.device)
                        ema_vae.step(vae.parameters())
                    train_state.global_step += 1
                    logs["global_step"] = train_state.global_step
                    train_loss = 0.0

                logs["lr"] = optimizer.param_groups[0]['lr']
                logs['epoch'] = epoch
                logs.update(log_dict)

                accelerator.log(logs)
                progress_bar.set_postfix(**logs)

        train_state.epoch += 1
        if epoch % args.eval_epochs == 0:
            with torch.no_grad():
                eval_loss, eval_log_dict = validation(eval_loader, vae, accelerator, weight_dtype, loss_fn,
                                                      train_state.global_step)
                lr_scheduler.step(eval_loss)

                if args.use_ema:
                    ema_vae.store(vae.parameters())
                    ema_vae.copy_to(vae.parameters())
                    _ = validation(eval_loader, vae, accelerator, weight_dtype, loss_fn, train_state.global_step, 'ema')
                    ema_vae.restore(vae.parameters())

            if accelerator.is_main_process:
                logger.info(f"Epoch {epoch} - LOSS: {eval_loss}")
                accelerator.save_state()

        if accelerator.is_main_process and epoch % args.model_save_interval == 0:
            vae.save_pretrained(args.output_dir / f"model_{epoch:04d}")

    if accelerator.is_main_process:
        vae = accelerator.unwrap_model(vae)
        vae.save_pretrained(args.output_dir)

        if args.use_ema:
            ema_vae.copy_to(vae.parameters())
            vae.save_pretrained(args.output_dir / f"ema")

    accelerator.end_training()
    logger.info("***** Training finished *****")


if __name__ == "__main__":
    train()
