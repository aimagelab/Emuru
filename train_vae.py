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

from models.htr import HTR
from models.autoencoder_kl import AutoencoderKL
from transformers.optimization import get_scheduler
from diffusers.training_utils import EMAModel

from utils import TrainState
from custom_datasets import OnlineFontSquare, TextSampler, collate_fn
from models.autoencoder_loss import AutoencoderLoss
from custom_datasets.font_square.font_square import make_renderers

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__)


@torch.no_grad()
def validation(eval_loader, vae, accelerator, loss_fn, weight_dtype, htr, writer_id, wandb_prefix="eval"):
    vae_model = accelerator.unwrap_model(vae)
    vae_model.eval()
    htr_model = accelerator.unwrap_model(htr)
    htr_model.eval()
    writer_id_model = accelerator.unwrap_model(writer_id)
    writer_id_model.eval()
    eval_loss = 0.
    images_for_log = []
    images_for_log_w_htr_wid = []

    for step, batch in enumerate(eval_loader):
        with accelerator.autocast():
            images = batch['images'].to(weight_dtype)
            targets_rgba_texts = batch['rgba_texts'].to(weight_dtype)
            writers = batch['writers']

            text_logits_s2s = batch['text_logits_s2s']
            text_logits_s2s_unpadded_len = batch['unpadded_texts_len']
            tgt_mask = batch['tgt_key_mask']
            tgt_key_padding_mask = batch['tgt_key_padding_mask']

            posterior = vae_model.encode(images).latent_dist
            z = posterior.sample()
            pred = vae_model.decode(z).sample

            loss, log_dict, wandb_media_log = loss_fn(images=targets_rgba_texts, z=z, reconstructions=pred, posteriors=posterior,
                                                      writers=writers, text_logits_s2s=text_logits_s2s,
                                                      text_logits_s2s_length=text_logits_s2s_unpadded_len,
                                                      tgt_key_padding_mask=tgt_key_padding_mask, source_mask=tgt_mask,
                                                      split=wandb_prefix, htr=htr_model, writer_id=writer_id_model)

            eval_loss += loss['loss'].item()

            if step == 0:
                alpha_fake = torch.ones_like(images[:, :1])
                rgba_images = torch.cat([images, alpha_fake], dim=1)
                images_for_log.append(torch.cat([rgba_images.cpu(), pred.cpu()], dim=-1)[:8])

            if step < 2:
                author_id = batch['writers'][0].item()
                pred_author_id = wandb_media_log[f'{wandb_prefix}/predicted_authors'][0][0]
                text = batch['texts'][0]
                pred_text = wandb_media_log[f'{wandb_prefix}/predicted_characters'][0][0]

                alpha_fake = torch.ones_like(images[:, :1])
                rgba_images = torch.cat([images, alpha_fake], dim=1)

                images_for_log_w_htr_wid.append(
                    wandb.Image(
                    torch.cat([rgba_images.cpu()[0], pred.cpu()[0]], dim=-1),
                    caption=f'AID: {author_id}, Pred AID: {pred_author_id}, Text: {text}, Pred Text: {pred_text}')
                )

    grid_nrows = 2
    accelerator.log({
        **log_dict,
        f"{wandb_prefix}/loss": eval_loss / len(eval_loader),
        "Original (left), Reconstruction (right)":
            [wandb.Image(torchvision.utils.make_grid(image, nrow=grid_nrows, normalize=True, value_range=(-1, 1)))
             for _, image in enumerate(images_for_log)],
        "Image, HTR and Writer ID": images_for_log_w_htr_wid
    })

    del vae_model
    torch.cuda.empty_cache()
    return eval_loss / len(eval_loader)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='results_vae', help="output directory")
    parser.add_argument("--logging_dir", type=str, default='results_vae', help="logging directory")
    parser.add_argument("--train_batch_size", type=int, default=32, help="train batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="eval batch size")
    parser.add_argument("--epochs", type=int, default=10000, help="number of epochs to train the model")
    parser.add_argument("--eval_epochs", type=int, default=1, help="eval interval")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--seed", type=int, default=24, help="random seed")
    parser.add_argument('--wandb_log_interval_steps', type=int, default=25, help="model save interval")
    parser.add_argument("--resume_id", type=str, default=None, help="resume from checkpoint")
    parser.add_argument("--vae_config", type=str, default='configs/vae/VAE_64x768.json', help='vae config path')
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--wandb_project_name", type=str, default="emuru_vae", help="wandb project name")

    parser.add_argument("--htr_path", type=str, default='results_htr/htr0', help='htr checkpoint path')
    parser.add_argument("--writer_id_path", type=str, default='results_writer_id/wd0/model_0125', help='writerid config path')
    parser.add_argument("--use_old_writer_id", type=str, default="False")
    
    parser.add_argument("--num_samples_per_epoch", type=int, default=None)
    parser.add_argument("--lr_scheduler", type=str, default="reduce_lr_on_plateau")
    parser.add_argument("--lr_scheduler_patience", type=int, default=10)
    parser.add_argument("--use_ema", type=str, default="False")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="no")
    parser.add_argument("--checkpoints_total_limit", type=int, default=5)

    parser.add_argument("--load_font_into_mem", type=str, default="True")

    args = parser.parse_args()

    args.use_ema = args.use_ema == "True"
    args.load_font_into_mem = args.load_font_into_mem == "True"
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

    
    if args.use_old_writer_id:
        logger.info("Using old writer id model implementation")
        from models.writer_id_old import WriterID
    else:
        from models.writer_id import WriterID

    vae = AutoencoderKL.from_config(args.vae_config)
    vae.train()
    vae.requires_grad_(True)
    args.vae_params = vae.num_parameters(only_trainable=True)

    if args.use_ema:
        ema_vae = vae.from_config(args.vae_config)
        ema_vae = EMAModel(ema_vae.parameters(), model_cls=AutoencoderKL, model_config=vae.config)
        accelerator.register_for_checkpointing(ema_vae)

    optimizer = torch.optim.AdamW(
        vae.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon)

    text_sampler = TextSampler(8, 32, (4, 7), exponent=0.5)
    renderers = make_renderers('files/font_square/clean_fonts', calib_threshold=0.8, verbose=True, load_font_into_mem=args.load_font_into_mem)
    train_dataset = OnlineFontSquare('files/font_square/clean_fonts', 'files/font_square/backgrounds',
                                     text_sampler=text_sampler, length=args.num_samples_per_epoch, load_font_into_mem=args.load_font_into_mem, 
                                     renderers=renderers)
    eval_dataset = OnlineFontSquare('files/font_square/clean_fonts', 'files/font_square/backgrounds',
                                    text_sampler=text_sampler, length=args.num_samples_per_epoch, load_font_into_mem=args.load_font_into_mem, 
                                    renderers=renderers)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=4, persistent_workers=True)
    eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=4, persistent_workers=True)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        scheduler_specific_kwargs={"patience": args.lr_scheduler_patience}
    )

    htr = HTR.from_pretrained(args.htr_path)
    writer_id = WriterID.from_pretrained(args.writer_id_path)
    htr.eval()
    writer_id.eval()
    for param in htr.parameters():
        param.requires_grad = False
    for param in writer_id.parameters():
        param.requires_grad = False

    loss_fn = AutoencoderLoss(alphabet=train_dataset.alphabet)
    args.htr_params = sum([p.numel() for p in htr.parameters()])
    args.writer_id_params = sum([p.numel() for p in writer_id.parameters()])
    args.total_params = args.vae_params + args.htr_params + args.writer_id_params

    vae, htr, writer_id, optimizer, train_loader, eval_loader, lr_scheduler, loss_fn = accelerator.prepare(
        vae, htr, writer_id, optimizer, train_loader, eval_loader, lr_scheduler, loss_fn)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if accelerator.is_main_process:
        wandb_args = {"wandb": {"entity": "fomo_aiisdh", "name": args.run_name}}
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.wandb_project_name, tracker_config, wandb_args)
        wandb.watch(vae, log="all", log_freq=1)

    num_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.epochs * num_steps_per_epoch
    total_batch_size = (args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps)

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}. Num steps per epoch = {num_steps_per_epoch}")
    logger.info(f"  Num eval samples = {len(eval_dataset)}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total trainable parameters count = {args.vae_params}. VAE: {args.vae_params}, HTR: {args.htr_params}, WriterID: {args.writer_id_params}")

    train_state = TrainState(global_step=0, epoch=0, best_eval_init=float('inf'))
    accelerator.register_for_checkpointing(train_state)
    if args.resume_id:
        try:
            accelerator.load_state()
            accelerator.project_configuration.iteration = train_state.epoch
            if train_state.best_eval == 0.0:
                train_state.best_eval = float('inf')
        except FileNotFoundError as e:
            logger.info(f"Checkpoint not found: {e}. Creating a new run")

    progress_bar = tqdm(range(train_state.global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(train_state.epoch, args.epochs):
        vae.train()
        train_loss = 0.

        for _, batch in enumerate(train_loader):

            with accelerator.autocast():
                with accelerator.accumulate(vae):
                    images = batch['images'].to(weight_dtype)
                    targets_rgba_texts = batch['rgba_texts'].to(weight_dtype)
                    writers = batch['writers']

                    text_logits_s2s = batch['text_logits_s2s']
                    text_logits_s2s_unpadded_len = batch['unpadded_texts_len']
                    tgt_mask = batch['tgt_key_mask']
                    tgt_key_padding_mask = batch['tgt_key_padding_mask']

                    if accelerator.num_processes > 1:
                        posterior = vae.module.encode(images).latent_dist
                    else:
                        posterior = vae.encode(images).latent_dist
                    z = posterior.sample()
                    if accelerator.num_processes > 1:
                        pred = vae.module.decode(z).sample
                    else:
                        pred = vae.decode(z).sample

                    loss, log_dict, _ = loss_fn(images=targets_rgba_texts, z=z, reconstructions=pred, posteriors=posterior,
                                                writers=writers, text_logits_s2s=text_logits_s2s,
                                                text_logits_s2s_length=text_logits_s2s_unpadded_len,
                                                tgt_key_padding_mask=tgt_key_padding_mask, source_mask=tgt_mask,
                                                split="train", htr=htr, writer_id=writer_id)
                    
                    loss = loss['loss']

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

                progress_bar.set_postfix(**logs)
                if train_state.global_step % args.wandb_log_interval_steps == 0:
                    accelerator.log(logs)

        train_state.epoch += 1
        if epoch % args.eval_epochs == 0 and accelerator.is_main_process:
            with torch.no_grad():
                eval_loss = validation(eval_loader, vae, accelerator, loss_fn, weight_dtype,  htr, writer_id, 'eval')
                eval_loss = broadcast(torch.tensor(eval_loss, device=accelerator.device), from_process=0)

                if args.use_ema:
                    ema_vae.store(vae.parameters())
                    ema_vae.copy_to(vae.parameters())
                    _ = validation(eval_loader, vae, accelerator, loss_fn, weight_dtype,  htr, writer_id, 'ema')
                    ema_vae.restore(vae.parameters())

                if eval_loss < train_state.best_eval:
                    train_state.best_eval = eval_loss.item()
                    vae_model = accelerator.unwrap_model(vae)
                    vae_model.save_pretrained(args.output_dir / f"model_{epoch:04d}")
                    del vae_model
                    logger.info(f"Epoch {epoch} - Best eval loss: {eval_loss}")

                train_state.last_eval = eval_loss.item()

            accelerator.save_state()

        lr_scheduler.step(eval_loss)
        accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()
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
