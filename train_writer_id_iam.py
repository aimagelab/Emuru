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
from accelerate.utils import broadcast
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed

from transformers.optimization import get_scheduler
import evaluate
import torchvision

from utils import TrainState
from custom_datasets import OnlineFontSquare, TextSampler, collate_fn
from custom_datasets.constants import END_OF_SEQUENCE
from custom_datasets.font_square.font_square import make_renderers
from models.writer_id import WriterID
from custom_datasets import dataset_factory
from custom_datasets.constants import FONT_SQUARE_CHARSET, PAD, START_OF_SEQUENCE, END_OF_SEQUENCE
from custom_datasets.alphabet import Alphabet
from custom_datasets.font_square import pad_sequence, subsequent_mask

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__)


@torch.no_grad()
def validation(eval_loader, writer_id, accelerator, weight_dtype, loss_fn, accuracy_fn, wandb_prefix="eval"):
    writer_id_model = accelerator.unwrap_model(writer_id)
    writer_id_model.eval()
    eval_loss = 0.
    images_for_log = []

    for step, batch in enumerate(eval_loader):
        images = batch['style_img'][:, :1, :, :768].to(weight_dtype)  # keep only one channel to have greyscale
        _, _, _, w = images.shape
        if w < 768:
                images = torchvision.functional.pad(images, (0, 0, 768 - w, 0), fill=1.)
        authors_id = batch['style_author_idx'].to(torch.int64)

        output = writer_id_model(images)

        loss = loss_fn(output, authors_id)
        predicted_authors = torch.argmax(output, dim=1)

        accuracy_fn.add_batch(predictions=predicted_authors.int(), references=authors_id.int())
        eval_loss += loss.item()

        if step == 0:
            images_for_log.append(wandb.Image(images[0], caption=f'Real: {authors_id.int()[0]}. '
                                                                 f'Pred: {predicted_authors.int()[0]}'))

    accuracy_value = accuracy_fn.compute()['accuracy']

    accelerator.log({
        f"{wandb_prefix}/loss": eval_loss / len(eval_loader),
        f"{wandb_prefix}/accuracy": accuracy_value,
        f"{wandb_prefix}/images": images_for_log,
    })

    del writer_id_model
    del images_for_log
    torch.cuda.empty_cache()
    return accuracy_value


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default='results_wid_iam', help="output directory")
    parser.add_argument("--logging_dir", type=str, default='results_wid_iam', help="logging directory")
    parser.add_argument("--train_batch_size", type=int, default=64, help="train batch size")
    parser.add_argument("--eval_batch_size", type=int, default=512, help="eval batch size")
    parser.add_argument("--epochs", type=int, default=10000, help="number of train epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--seed", type=int, default=24, help="random seed")
    parser.add_argument('--wandb_log_interval_steps', type=int, default=1, help="model save interval")
    parser.add_argument("--eval_epochs", type=int, default=1, help="eval interval")
    parser.add_argument("--resume_id", type=str, default=None, help="resume from checkpoint")
    parser.add_argument("--run_id", type=str, default=uuid.uuid4().hex[:4], help="uuid of the run")
    parser.add_argument("--writer_id_config", type=str, default='configs/writer_id/WriterID_64x768.json', help='config path')
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--wandb_project_name", type=str, default="emuru_writer_id_iam", help="wandb project name")

    parser.add_argument("--num_samples_per_epoch", type=int, default=None)
    parser.add_argument("--lr_scheduler", type=str, default="reduce_lr_on_plateau")
    parser.add_argument("--lr_scheduler_patience", type=int, default=5)
    parser.add_argument("--use_ema", type=str, default="False")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="no")
    parser.add_argument("--checkpoints_total_limit", type=int, default=5)

    parser.add_argument("--load_font_into_mem", type=str, default="True")
    parser.add_argument("--load_font_num_threads", type=int, default=8)

    args = parser.parse_args()

    args.use_ema = args.use_ema == "True"
    args.load_font_into_mem = args.load_font_into_mem == "True"
    args.adam_beta1 = 0.9
    args.adam_beta2 = 0.999
    args.adam_epsilon = 1e-8
    args.adam_weight_decay = 0

    args.run_name = args.resume_id if args.resume_id else args.run_id
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

    with open(args.writer_id_config, "r") as f:
        config_dict = json.load(f)

    train_dataset =  dataset_factory('train', ['iam_lines'], root_path='/home/fquattrini/emuru/files/datasets/')
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=train_dataset.collate_fn, 
                              num_workers=4, persistent_workers=False)
    
    eval_dataset =  dataset_factory('test', ['iam_lines'], root_path='/home/fquattrini/emuru/files/datasets/')
    eval_loader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, shuffle=True, collate_fn=eval_dataset.collate_fn, 
                              num_workers=4, persistent_workers=False) 

    wid_path = 'results_wid/08be/model_0175'
    writer_id = WriterID.from_pretrained(wid_path)
    writer_id.reset_last_layer(len(set(train_dataset.authors)))
    # writer_id.requires_grad_(True)
    writer_id.requires_grad_(False)
    writer_id.linear2.requires_grad_(True)

    if args.use_ema:
        ema_writer_id = WriterID.from_config(config_dict)
        ema_writer_id = EMAModel(ema_writer_id.parameters(), model_cls=WriterID, model_config=writer_id.config)
        accelerator.register_for_checkpointing(ema_writer_id)

    optimizer = torch.optim.Adam(
        writer_id.parameters(),
        lr=args.lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        scheduler_specific_kwargs={"patience": args.lr_scheduler_patience, 'mode': 'max'}
    )

    writer_id, optimizer, train_loader, eval_loader, lr_scheduler = accelerator.prepare(writer_id, optimizer, train_loader, eval_loader, lr_scheduler)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    if accelerator.is_main_process:
        wandb_args = {"wandb": {"entity": "fomo_aiisdh", "name": args.run_name}}
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.wandb_project_name, tracker_config, wandb_args)
        wandb.watch(writer_id, log="all", log_freq=1000)

    num_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.epochs * num_steps_per_epoch
    total_batch_size = (args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps)
    args.total_params = sum([p.numel() for p in writer_id.parameters()])

    logger.info("***** Running HTR training *****")
    logger.info(f"  Num train samples = {len(train_dataset)}. Num steps per epoch = {num_steps_per_epoch}")
    logger.info(f"  Num eval samples = {len(eval_dataset)}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total parameters count = {args.total_params}")

    train_state = TrainState(global_step=0, epoch=0, best_eval_init=0.0)
    accelerator.register_for_checkpointing(train_state)
    if args.resume_id:
        try:
            accelerator.load_state()
            accelerator.project_configuration.iteration = train_state.epoch
        except FileNotFoundError as e:
            logger.info(f"Checkpoint not found: {e}. Creating a new run")
        
    ce_loss = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    accuracy = evaluate.load('accuracy')

    progress_bar = tqdm(range(train_state.global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    for epoch in range(train_state.epoch, args.epochs):

        writer_id.train()
        train_loss = 0.
        train_accuracy = 0.

        for _, batch in enumerate(train_loader):

            with accelerator.accumulate(writer_id):
                images = batch['style_img'][:, :1, :, :768].to(weight_dtype)  # keep only one channel to have greyscale
                _, _, _, w = images.shape
                if w < 768:
                     images = torchvision.functional.pad(images, (0, 0, 768 - w, 0), fill=1.)
                authors_id = batch['style_author_idx'].to(torch.int64)

                output = writer_id(images)

                loss = ce_loss(output, authors_id)
                predicted_authors = torch.argmax(output, dim=1)
                accuracy_value = accuracy.compute(predictions=predicted_authors.int(), references=authors_id.int())['accuracy']

                if not torch.isfinite(loss):
                    logger.info("\nWARNING: non-finite loss")
                    optimizer.zero_grad()
                    continue

                avg_loss = accelerator.gather(loss).mean()
                avg_accuracy = accelerator.gather(torch.tensor(accuracy_value).to(accelerator.device)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps
                train_accuracy += avg_accuracy.item() / args.gradient_accumulation_steps
                accelerator.backward(loss)

                optimizer.step()
                optimizer.zero_grad()

            logs = {}
            if accelerator.sync_gradients:
                progress_bar.update(1)
                if args.use_ema:
                    ema_writer_id.to(writer_id.device)
                    ema_writer_id.step(writer_id.parameters())

                train_state.global_step += 1
                logs["global_step"] = train_state.global_step
                logs['train/loss'] = train_loss
                logs['train/accuracy'] = train_accuracy
                train_loss = 0.0
                train_accuracy = 0.0

            logs["lr"] = optimizer.param_groups[0]['lr']
            logs["train/ce"] = loss.detach().item()
            logs["train/accuracy"] = accuracy_value
            logs['epoch'] = epoch

            progress_bar.set_postfix(**logs)
            if train_state.global_step % args.wandb_log_interval_steps == 0:
                accelerator.log(logs)

        train_state.epoch += 1

        if epoch % args.eval_epochs == 0:
            if accelerator.is_main_process:
                with torch.no_grad():
                    eval_accuracy = validation(eval_loader, writer_id, accelerator, weight_dtype, ce_loss, accuracy, 'eval')
                    eval_accuracy = broadcast(torch.tensor(eval_accuracy, device=accelerator.device), from_process=0)

                    if args.use_ema:
                        ema_writer_id.store(writer_id.parameters())
                        ema_writer_id.copy_to(writer_id.parameters())
                        _ = validation(eval_loader, writer_id, accelerator, weight_dtype, ce_loss, accuracy, 'ema')
                        ema_writer_id.restore(writer_id.parameters())

                    if eval_accuracy > train_state.best_eval:
                        train_state.best_eval = eval_accuracy
                        writer_id_to_save = accelerator.unwrap_model(writer_id)
                        writer_id_to_save.save_pretrained(args.output_dir / f"model_{epoch:04d}")
                        del writer_id_to_save
                        logger.info(f"Epoch {epoch} - Best eval accuracy: {eval_accuracy}")
                
                train_state.last_eval = eval_accuracy
                accelerator.save_state()

            accelerator.wait_for_everyone()
            lr_scheduler.step(train_state.last_eval)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        writer_id = accelerator.unwrap_model(writer_id)
        writer_id.save_pretrained(args.output_dir)

        if args.use_ema:
            ema_writer_id.copy_to(writer_id.parameters())
            writer_id.save_pretrained(args.output_dir / f"ema")

    accelerator.end_training()
    logger.info("***** Training finished *****")


if __name__ == "__main__":
    train()