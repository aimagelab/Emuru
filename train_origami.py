from models import OrigamiNet, CTCLabelConverter
import torch
import copy
from transformers import AutoTokenizer
from transformers import T5ForConditionalGeneration, T5Config
from transformers.models.t5.modeling_t5 import T5Stack
from diffusers import AutoencoderKL
from custom_datasets import OnlineFontSquare, HFDataCollector, TextSampler, FixedTextSampler
from einops.layers.torch import Rearrange
from einops import repeat
from torch.nn import MSELoss
from pathlib import Path
from torch.utils.data import DataLoader
import wandb
import argparse
from tqdm import tqdm
from utils import MetricCollector
from torchvision.utils import make_grid

from models import AutoencoderKL as LightningAutoencoderKL
import torch.nn.init as init

def init_bn(model):
    if type(model) in [torch.nn.InstanceNorm2d, torch.nn.BatchNorm2d]:
        init.ones_(model.weight)
        init.zeros_(model.bias)

    elif type(model) in [torch.nn.Conv2d]:
        init.kaiming_uniform_(model.weight)


def train(args):
    text_sampler = TextSampler(8, 32, 4)
    dataset = OnlineFontSquare('files/font_square/fonts', 'files/font_square/backgrounds', text_sampler)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn, drop_last=True,
                        num_workers=args.dataloader_num_workers, persistent_workers=args.dataloader_num_workers > 0)

    model = OrigamiNet(len(text_sampler.charset) + 1, n_channels=1)
    model.apply(init_bn)
    model.to(args.device)

    text_encoder = CTCLabelConverter(text_sampler.charset).to(args.device)
    vae = LightningAutoencoderKL.load_from_checkpoint(args.vae_checkpoint, strict=False).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=10**(-1/90000))
    criterion = torch.nn.CTCLoss(reduction='mean', zero_infinity=True)

    if args.resume:
        checkpoint = torch.load(Path(args.output_dir) / 'origami.pth')
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch']
            args = checkpoint['args']
        else:
            model.load_state_dict(checkpoint)
        print(f'Resuming training from epoch {args.start_epoch}')

    # eval_fonts = sorted(Path('files/font_square/fonts').rglob('*.ttf'))[:100]
    # dataset_eval = OnlineFontSquare(eval_fonts, [], FixedTextSampler('this is a test'))
    # loader_eval = DataLoader(dataset_eval, batch_size=args.batch_size, shuffle=False, collate_fn=model.data_collator, num_workers=args.dataloader_num_workers)

    if args.wandb: wandb.init(project='Emuru_origami', config=args)
    collector = MetricCollector()

    for epoch in range(args.start_epoch, args.num_train_epochs):
        model.train()
        for i, batch in tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch}'):
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            enc_text, enc_text_len = text_encoder.encode(batch['text'], args.device)

            with torch.no_grad():
                posterior = vae.encode(batch['bw_img'].float())
                z = posterior.sample()
            preds = model(z)
            # preds = model(batch['bw_img'].float())

            preds_len = torch.IntTensor([preds.size(1)] * args.batch_size).to(args.device)
            preds = preds.permute(1, 0, 2).log_softmax(2)

            torch.backends.cudnn.enabled = False
            loss = criterion(preds, enc_text, preds_len, enc_text_len)
            torch.backends.cudnn.enabled = True

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            collector['ctc_loss'] = loss.item()

        with torch.no_grad():
            model.eval()
            dec_text = text_encoder.decode(preds.permute(1, 0, 2).argmax(-1), preds_len)[:4]
            for gt, txt in zip(batch['text'], dec_text):
                print(gt)
                print(txt)
            if args.wandb: wandb.log({
                'lr': lr_scheduler.get_lr()[0],
                'sample_img': wandb.Image(batch['bw_img'][0], caption=f'{batch["text"][0]}\n{dec_text[0]}'),
            } | collector.dict())
                
        if epoch % 10 == 0 and epoch > 0:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args,
            }, Path(args.output_dir) / 'origami.pth')
            print(f'Saved model at epoch {epoch} in {args.output_dir}')
        
        collector.reset()
        lr_scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a T5 model with a VAE')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--t5_checkpoint', type=str, default='google-t5/t5-small', help='T5 checkpoint')
    parser.add_argument('--vae_checkpoint', type=str, default='files/checkpoints/lightning_vae/vae.ckpt', help='VAE checkpoint')
    parser.add_argument('--output_dir', type=str, default='files/checkpoints/Emuru_sm', help='Output directory')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--start_epoch', type=int, default=0, help='Start epoch')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--num_train_epochs', type=int, default=10 ** 10, help='Number of train epochs')
    parser.add_argument('--report_to', type=str, default='none', help='Report to')
    parser.add_argument('--dataloader_num_workers', type=int, default=12, help='Dataloader num workers')
    parser.add_argument('--slices_per_query', type=int, default=1, help='Number of slices to predict in each query')
    parser.add_argument('--wandb', action='store_true', help='Use wandb')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    args = parser.parse_args()

    train(args)