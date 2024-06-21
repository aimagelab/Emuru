import argparse
import logging
import wandb
from pathlib import Path
import math
from tqdm.auto import tqdm
import uuid
import json
import gc

import torch
from torch.utils.data import DataLoader
import torch.utils.checkpoint
import torchvision

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.utils import broadcast

from models.htr import HTR
from models.writer_id import WriterID
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

torch.autograd.set_detect_anomaly(True)

@torch.no_grad()
def validation(eval_loader, vae, accelerator, loss_fn, weight_dtype, htr, writer_id, wandb_prefix="eval"):
    vae_model = accelerator.unwrap_model(vae)
    vae_model.eval()
    htr_model = accelerator.unwrap_model(htr)
    htr_model.eval()
    writer_id_model = accelerator.unwrap_model(writer_id)
    writer_id_model.eval()
    eval_loss_vae = 0.
    eval_loss_htr = 0.
    eval_loss_writer_id = 0.
    images_for_log = []
    images_for_log_w_htr_wid = []

    for step, batch in enumerate(eval_loader):
        with accelerator.autocast():
            images = batch['images'].to(weight_dtype)
            targets = batch['images_bw'].to(weight_dtype)
            writers = batch['writers']

            text_logits_s2s = batch['text_logits_s2s']
            text_logits_s2s_unpadded_len = batch['unpadded_texts_len']
            tgt_mask = batch['tgt_key_mask']
            tgt_key_padding_mask = batch['tgt_key_padding_mask']

            posterior = vae_model.encode(images).latent_dist
            z = posterior.sample()
            pred = vae_model.decode(z).sample

            loss, log_dict, wandb_media_log = loss_fn(images=targets, z=z, reconstructions=pred, posteriors=posterior,
                                                      writers=writers, text_logits_s2s=text_logits_s2s,
                                                      text_logits_s2s_length=text_logits_s2s_unpadded_len,
                                                      tgt_key_padding_mask=tgt_key_padding_mask, source_mask=tgt_mask,
                                                      split=wandb_prefix, htr=htr_model, writer_id=writer_id_model)
            
            loss_vae = loss['loss']
            loss_htr = loss['htr_loss']
            loss_writer_id = loss['writer_loss']

            eval_loss_vae += loss_vae.item()
            eval_loss_htr += loss_htr.item()
            eval_loss_writer_id += loss_writer_id.item()

            if step == 0:
                images_for_log.append(torch.cat([images.cpu(), pred.repeat(1, 3, 1, 1).cpu()], dim=-1)[:8])

            if step < 2:
                author_id = batch['writers'][0].item()
                pred_author_id = wandb_media_log[f'{wandb_prefix}/predicted_authors'][0][0]
                text = batch['texts'][0]
                pred_text = wandb_media_log[f'{wandb_prefix}/predicted_characters'][0][0]
                images_for_log_w_htr_wid.append(wandb.Image(
                    torch.cat([images.cpu()[0], pred.repeat(1, 3, 1, 1).cpu()[0]], dim=-1),
                    caption=f'AID: {author_id}, Pred AID: {pred_author_id}, Text: {text}, Pred Text: {pred_text}')
                )

    grid_nrows = 2
    accelerator.log({
        **log_dict,
        f"{wandb_prefix}/loss_vae": loss_vae / len(eval_loader),
        f"{wandb_prefix}/loss_htr": loss_htr / len(eval_loader),
        f"{wandb_prefix}/loss_writer_id": loss_writer_id / len(eval_loader),
        "Original (left), Reconstruction (right)":
            [wandb.Image(torchvision.utils.make_grid(image, nrow=grid_nrows, normalize=True, value_range=(-1, 1)))
             for _, image in enumerate(images_for_log)],
        "Image, HTR and Writer ID": images_for_log_w_htr_wid
    })

    del vae_model
    torch.cuda.empty_cache()
    return eval_loss_vae / len(eval_loader), eval_loss_htr / len(eval_loader), eval_loss_writer_id / len(eval_loader)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='results_vae', help="output directory")
    parser.add_argument("--logging_dir", type=str, default='results_vae', help="logging directory")
    parser.add_argument("--train_batch_size", type=int, default=4, help="train batch size")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="eval batch size")
    parser.add_argument("--epochs", type=int, default=10000, help="number of epochs to train the model")
    parser.add_argument("--eval_epochs", type=int, default=5, help="eval interval")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--seed", type=int, default=24, help="random seed")
    parser.add_argument('--wandb_log_interval_steps', type=int, default=25, help="model save interval")
    parser.add_argument("--resume_id", type=str, default=None, help="resume from checkpoint")
    parser.add_argument("--vae_config", type=str, default='configs/vae/VAE_64x768.json', help='vae config path')
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--wandb_project_name", type=str, default="emuru_vae_latent", help="wandb project name")

    parser.add_argument("--htr_config", type=str, default='configs/htr/HTR_64x768_latent.json', help='config path')
    parser.add_argument("--writer_id_config", type=str, default='configs/writer_id/WriterID_64x768_latent.json', help='config path')
    
    parser.add_argument("--num_samples_per_epoch", type=int, default=None)
    parser.add_argument("--lr_scheduler", type=str, default="reduce_lr_on_plateau")
    parser.add_argument("--lr_scheduler_patience", type=int, default=5)
    parser.add_argument("--use_ema", type=str, default="False")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
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

    vae = AutoencoderKL.from_config(args.vae_config)
    vae.train()
    vae.requires_grad_(True)
    args.vae_params = vae.num_parameters(only_trainable=True)

    if args.use_ema:
        ema_vae = vae.from_config(args.vae_config)
        ema_vae = EMAModel(ema_vae.parameters(), model_cls=AutoencoderKL, model_config=vae.config)
        accelerator.register_for_checkpointing(ema_vae)

    optimizer_vae = torch.optim.AdamW(
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

    lr_scheduler_vae = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_vae,
        scheduler_specific_kwargs={"patience": args.lr_scheduler_patience}
    )

    htr = HTR.from_config(args.htr_config)
    writer_id = WriterID.from_config(args.writer_id_config)
    htr.train()
    writer_id.train()

    optimizer_htr = torch.optim.AdamW(
        htr.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon)
    
    optimizer_writer_id = torch.optim.AdamW(
        writer_id.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon)
    
    lr_scheduler_htr = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_htr,
        scheduler_specific_kwargs={"patience": args.lr_scheduler_patience, 'mode': 'max'}
    )

    lr_scheduler_writer_id = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer_writer_id,
        scheduler_specific_kwargs={"patience": args.lr_scheduler_patience}
    )

    loss_fn = AutoencoderLoss(alphabet=train_dataset.alphabet)
    args.htr_params = sum([p.numel() for p in htr.parameters()])
    args.writer_id_params = sum([p.numel() for p in writer_id.parameters()])
    args.total_params = args.vae_params + args.htr_params + args.writer_id_params

    vae, htr, writer_id, loss_fn = accelerator.prepare(vae, htr, writer_id, loss_fn)
    optimizer_vae, optimizer_htr, optimizer_writer_id = accelerator.prepare(optimizer_vae, optimizer_htr, optimizer_writer_id)
    lr_scheduler_vae, lr_scheduler_htr, lr_scheduler_writer_id = accelerator.prepare(lr_scheduler_vae, lr_scheduler_htr, lr_scheduler_writer_id)
    train_loader, eval_loader = accelerator.prepare(train_loader, eval_loader)

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
    logger.info(f"  Total parameters count = {args.total_params}. VAE: {args.vae_params}, HTR: {args.htr_params}, WriterID: {args.writer_id_params}")

    train_state = TrainState(global_step=0, epoch=0)
    accelerator.register_for_checkpointing(train_state)
    if args.resume_id:
        try:
            accelerator.load_state()
            accelerator.project_configuration.iteration = train_state.epoch
        except FileNotFoundError as e:
            logger.info(f"Checkpoint not found: {e}. Creating a new run")

    progress_bar = tqdm(range(train_state.global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(train_state.epoch, args.epochs):
        vae.train()
        htr.train()
        writer_id.train()
        train_loss_vae = 0.
        train_loss_htr = 0.
        train_loss_writer_id = 0.

        for _, batch in enumerate(train_loader):

            with accelerator.autocast():
                with accelerator.accumulate(vae):
                    images = batch['images'].to(weight_dtype)
                    targets = batch['images_bw'].to(weight_dtype)
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

                    loss, log_dict, _ = loss_fn(images=targets, z=z, reconstructions=pred, posteriors=posterior,
                                                writers=writers, text_logits_s2s=text_logits_s2s,
                                                text_logits_s2s_length=text_logits_s2s_unpadded_len,
                                                tgt_key_padding_mask=tgt_key_padding_mask, source_mask=tgt_mask,
                                                split="train", htr=htr, writer_id=writer_id)
                    
                    loss_vae = loss['loss']
                    loss_htr = loss['htr_loss']
                    loss_writer_id = loss['writer_loss']

                    if not torch.isfinite(loss_vae) or not torch.isfinite(loss_htr) or not torch.isfinite(loss_writer_id):
                        logger.info("\nWARNING: non-finite loss")
                        optimizer_vae.zero_grad()
                        optimizer_htr.zero_grad()
                        optimizer_writer_id.zero_grad()
                        continue

                    avg_loss_vae = accelerator.gather(loss_vae).mean()
                    avg_loss_htr = accelerator.gather(loss_htr).mean()
                    avg_loss_writer_id = accelerator.gather(loss_writer_id).mean()
                    train_loss_vae += avg_loss_vae.item() / args.gradient_accumulation_steps
                    train_loss_htr += avg_loss_htr.item() / args.gradient_accumulation_steps
                    train_loss_writer_id += avg_loss_writer_id.item() / args.gradient_accumulation_steps
                    
                    optimizer_vae.zero_grad()
                    optimizer_htr.zero_grad()
                    optimizer_writer_id.zero_grad()

                    accelerator.backward(loss_vae)  # This should work because the vae loss contains htr and wid
                    optimizer_vae.step()
                    optimizer_htr.step()
                    optimizer_writer_id.step()

                logs = {}
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    if args.use_ema:
                        ema_vae.to(vae.device)
                        ema_vae.step(vae.parameters())
                    train_state.global_step += 1
                    logs["global_step"] = train_state.global_step
                    train_loss_vae = 0.0
                    train_loss_htr = 0.0
                    train_loss_writer_id = 0.0

                logs["lr_vae"] = optimizer_vae.param_groups[0]['lr']
                logs["lr_htr"] = optimizer_htr.param_groups[0]['lr']
                logs["lr_writer_id"] = optimizer_writer_id.param_groups[0]['lr']
                logs['epoch'] = epoch
                logs.update(log_dict)

                progress_bar.set_postfix(**logs)
                if train_state.global_step % args.wandb_log_interval_steps == 0:
                    accelerator.log(logs)

        train_state.epoch += 1
        if epoch % args.eval_epochs == 0:
            if accelerator.is_main_process:
                with torch.no_grad():
                    eval_loss_vae, eval_loss_htr, eval_loss_writer_id = validation(
                        eval_loader, vae, accelerator, loss_fn, weight_dtype, htr, writer_id, 'eval')
                    
                    eval_loss_vae = broadcast(torch.tensor(eval_loss_vae, device=accelerator.device), from_process=0)
                    eval_loss_htr = broadcast(torch.tensor(eval_loss_htr, device=accelerator.device), from_process=0)
                    eval_loss_writer_id = broadcast(torch.tensor(eval_loss_writer_id, device=accelerator.device), from_process=0)

                    if args.use_ema:
                        ema_vae.store(vae.parameters())
                        ema_vae.copy_to(vae.parameters())
                        _ = validation(eval_loader, vae, accelerator, loss_fn, weight_dtype, 'ema')
                        ema_vae.restore(vae.parameters())

                    if eval_loss_vae < train_state.best_eval:
                        train_state.best_eval = eval_loss_vae.item()
                        vae_model = accelerator.unwrap_model(vae)
                        vae_model.save_pretrained(args.output_dir / f"model_{epoch:04d}")
                        del vae_model
                        htr_model = accelerator.unwrap_model(htr)
                        htr_model.save_pretrained(args.output_dir / f"htr_{epoch:04d}")
                        del htr_model
                        writer_id_model = accelerator.unwrap_model(writer_id)
                        writer_id_model.save_pretrained(args.output_dir / f"writer_id_{epoch:04d}")
                        del writer_id_model
                        logger.info(f"Epoch {epoch} - Best eval loss: {eval_loss_vae}")

                train_state.last_eval = eval_loss_vae.item()
                accelerator.save_state()

            accelerator.wait_for_everyone()
            lr_scheduler_vae.step(eval_loss_vae)
            lr_scheduler_htr.step(eval_loss_htr)
            lr_scheduler_writer_id.step(eval_loss_writer_id)

        gc.collect()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        vae = accelerator.unwrap_model(vae)
        vae.save_pretrained(args.output_dir)

        htr = accelerator.unwrap_model(htr)
        htr.save_pretrained(args.output_dir)

        writer_id = accelerator.unwrap_model(writer_id)
        writer_id.save_pretrained(args.output_dir)

        if args.use_ema:
            ema_vae.copy_to(vae.parameters())
            vae.save_pretrained(args.output_dir / f"ema")

    accelerator.end_training()
    logger.info("***** Training finished *****")


if __name__ == "__main__":
    train()
