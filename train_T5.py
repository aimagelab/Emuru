import torch
import copy
from transformers import AutoTokenizer
from transformers import T5ForConditionalGeneration, T5Config
from custom_datasets import OnlineFontSquare, HFDataCollector, TextSampler, FixedTextSampler
from einops.layers.torch import Rearrange
from einops import repeat
from torch.nn import MSELoss, CTCLoss
from pathlib import Path
from torch.utils.data import DataLoader
import wandb
import argparse
from tqdm import tqdm
from utils import MetricCollector
from torchvision.utils import make_grid

from models import AutoencoderKL as LightningAutoencoderKL
from models import OrigamiNet


class Emuru(torch.nn.Module):
    def __init__(self,
                 t5_checkpoint="google-t5/t5-small",
                 vae_checkpoint='files/checkpoints/lightning_vae/vae.ckpt',
                 ocr_checkpoint='files/checkpoints/Origami_bw_img/origami.pth',
                 slices_per_query=1):
        super(Emuru, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')  # per-character tokenizer
        self.data_collator = HFDataCollector(tokenizer=self.tokenizer)

        config = T5Config.from_pretrained(t5_checkpoint)
        config.vocab_size = len(self.tokenizer)
        self.T5 = T5ForConditionalGeneration(config)
        self.T5.lm_head = torch.nn.Linear(config.d_model, 8 * slices_per_query, bias=False)
        self.sos = torch.nn.Embedding(1, config.d_model)
        self.query_emb = torch.nn.Linear(8 * slices_per_query, config.d_model)

        # self.vae = AutoencoderKL.from_pretrained(vae_checkpoint)
        self.vae = LightningAutoencoderKL.load_from_checkpoint(vae_checkpoint, strict=False)
        self.set_training(self.vae, False)

        self.ocr = OrigamiNet.from_checkpoint(ocr_checkpoint, o_classes=165, n_channels=1)
        self.set_training(self.ocr, False)
        
        self.query_rearrange = Rearrange('b c h (w q) -> b w (q c h)', q=slices_per_query)
        self.z_rearrange = Rearrange('b w (q c h) -> b c h (w q)', c=1, q=slices_per_query)
        self.z_rearrange_eval = Rearrange('w b (q c h) -> b c h (w q)', c=1, q=slices_per_query)

        self.mse_criterion = MSELoss()
        self.trainer = None

    def set_training(self, model, training):
        model.train() if training else model.eval()
        for param in model.parameters():
            param.requires_grad = training

    def _img_encode(self, img):
        posterior = self.vae.encode(img.float())
        z = posterior.sample()
        z_sequence = self.query_rearrange(z)
        decoder_inputs_embeds = self.query_emb(z_sequence)
        sos = repeat(self.sos.weight, '1 d -> b 1 d', b=decoder_inputs_embeds.size(0))
        decoder_inputs_embeds = torch.cat([sos, decoder_inputs_embeds], dim=1)
        return decoder_inputs_embeds, z_sequence, z

    def forward(self, text=None, img=None, input_ids=None, attention_mask=None, length=None):
        decoder_inputs_embeds, z_sequence, z = self._img_encode(img)
        output = self.T5(input_ids, attention_mask=attention_mask, decoder_inputs_embeds=decoder_inputs_embeds)

        mse_loss = self.mse_criterion(output.logits[:, :-1], z_sequence)

        pred_latent = self.z_rearrange(output.logits[:, :-1])
        ocr_gt_preds = self.ocr(img.float())
        ocr_img_preds = self.ocr(self.vae.decode(pred_latent))
        ocr_loss = self.mse_criterion(ocr_img_preds, ocr_gt_preds)

        alpha = 0.1
        loss = alpha * mse_loss + (1 - alpha) * ocr_loss
        return {'loss': loss, 'mse_loss': mse_loss, 'ocr_loss': ocr_loss}, pred_latent, z
    
    
    @torch.no_grad()
    def generate(self, text=None, img=None, input_ids=None, max_new_tokens=96, decoder_truncate=None):
        assert text is not None or input_ids is not None, 'Either text or input_ids must be provided'
        if input_ids is None:
            input_ids = self.tokenizer(text, return_tensors='pt').input_ids
            input_ids = input_ids.to(next(self.T5.parameters()).device)
        
        z_sequence = None
        if img is not None:
            _, z_sequence, _ = self._img_encode(img)

        if decoder_truncate is not None and z_sequence is not None:
            z_sequence = z_sequence[:, :decoder_truncate]

        new_z_sequence = [z_sequence, ] if z_sequence is not None else []
        sos = repeat(self.sos.weight, '1 d -> b 1 d', b=input_ids.size(0))
        for _ in range(max_new_tokens):
            if len(new_z_sequence) == 0:
                decoder_inputs_embeds = sos
            else:
                decoder_inputs_embeds = self.query_emb(torch.cat(new_z_sequence, dim=1))
                decoder_inputs_embeds = torch.cat([sos, decoder_inputs_embeds], dim=1)
            output = self.T5(input_ids, decoder_inputs_embeds=decoder_inputs_embeds)
            new_z_sequence.append(output.logits[:, -1:])

        z_sequence = torch.cat(new_z_sequence, dim=1)
        img = torch.clamp(self.vae.decode(self.z_rearrange(z_sequence)), -1, 1)
        return img
    
    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.T5.state_dict(), path / 'T5.pth')
        torch.save(self.vae.state_dict(), path / 'VAE.pth')
        torch.save(self.ocr.state_dict(), path / 'OCR.pth')
        torch.save(self.query_emb.state_dict(), path / 'query_emb.pth')
        torch.save(self.sos.state_dict(), path / 'sos.pth')

    def load_pretrained(self, path):
        path = Path(path)
        self.T5.load_state_dict(torch.load(path / 'T5.pth'))
        self.vae.load_state_dict(torch.load(path / 'VAE.pth'))
        self.ocr.load_state_dict(torch.load(path / 'OCR.pth'))
        self.query_emb.load_state_dict(torch.load(path / 'query_emb.pth'))
        self.sos.load_state_dict(torch.load(path / 'sos.pth'))


def train(args):
    if args.device == 'cpu':
        print('WARNING: Using CPU')

    model = Emuru(args.t5_checkpoint, args.vae_checkpoint, args.ocr_checkpoint, args.slices_per_query).to(args.device)

    if args.resume:
        model.load_pretrained(args.output_dir)
        model.to(args.device)
        print(f'Resumed training from {args.output_dir}')

    optimizer = torch.optim.AdamW(model.T5.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    dataset = OnlineFontSquare('files/font_square/fonts', 'files/font_square/backgrounds', TextSampler(8, 32, 4))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=model.data_collator,
                        num_workers=args.dataloader_num_workers, persistent_workers=args.dataloader_num_workers > 0)

    # eval_fonts = sorted(Path('files/font_square/fonts').rglob('*.ttf'))[:100]
    # dataset_eval = OnlineFontSquare(eval_fonts, [], FixedTextSampler('this is a test'))
    # loader_eval = DataLoader(dataset_eval, batch_size=args.batch_size, shuffle=False, collate_fn=model.data_collator, num_workers=args.dataloader_num_workers)

    if args.wandb: wandb.init(project='Emuru', config=args)
    collector = MetricCollector()

    for epoch in range(args.num_train_epochs):
        model.train()
        for i, batch in tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch}'):
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            losses, pred, gt = model(**batch)
            optimizer.zero_grad()
            losses['loss'].backward()
            optimizer.step()

            collector.update(losses)

            # print('Warning')
            # imgs = model.custom_generate(text='this is a sample text', img=None, max_new_tokens=96)
            # imgs = model.custom_generate(input_ids=batch['input_ids'], img=batch['img'], max_new_tokens=96 - 16, decoder_truncate=16)
            # print()

        # with torch.no_grad():
        #     model.eval()
        #     img = model.generate(text='this is a test')
        #     for i, batch in tqdm(enumerate(loader_eval), total=len(loader_eval), desc=f'Evaluating'):
        #         batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        #         img = model.generate(text='this is a test')

        with torch.no_grad():
            model.eval()
            pred = torch.clamp(model.vae.decode(pred), -1, 1)
            gt = torch.clamp(model.vae.decode(gt), -1, 1)
            img = torch.cat([batch['img'], gt, pred], dim=-1)[:16]

            def _continue_gen(style_len):
                test_img = model.generate(input_ids=batch['input_ids'], img=batch['img'], max_new_tokens=96 - style_len, decoder_truncate=style_len)[:16]
                test_img[:, :, :, style_len * 8] = -1  # add a black line between style and pred
                return test_img
            
            if args.wandb: wandb.log({
                'img_recon_pred': wandb.Image(make_grid(img, nrow=1, normalize=True)),
                'style_len_1': wandb.Image(make_grid(_continue_gen(1), nrow=1, normalize=True)),
                'style_len_8': wandb.Image(make_grid(_continue_gen(8), nrow=1, normalize=True)),
                'style_len_16': wandb.Image(make_grid(_continue_gen(16), nrow=1, normalize=True)),
                'style_len_32': wandb.Image(make_grid(_continue_gen(32), nrow=1, normalize=True)),
            } | collector.dict())
                
        if epoch % 10 == 0 and epoch > 0:
            model.save_pretrained(args.output_dir)
            print(f'Saved model at epoch {epoch} in {args.output_dir}')
        
        collector.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a T5 model with a VAE')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--t5_checkpoint', type=str, default='google-t5/t5-small', help='T5 checkpoint')
    parser.add_argument('--vae_checkpoint', type=str, default='files/checkpoints/lightning_vae/vae.ckpt', help='VAE checkpoint')
    parser.add_argument('--ocr_checkpoint', type=str, default='files/checkpoints/Origami_bw_img/origami.pth', help='OCR checkpoint')
    parser.add_argument('--output_dir', type=str, default='files/checkpoints/Emuru_sm', help='Output directory')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--num_train_epochs', type=int, default=10 ** 10, help='Number of train epochs')
    parser.add_argument('--report_to', type=str, default='none', help='Report to')
    parser.add_argument('--dataloader_num_workers', type=int, default=8, help='Dataloader num workers')
    parser.add_argument('--slices_per_query', type=int, default=1, help='Number of slices to predict in each query')
    parser.add_argument('--wandb', action='store_true', help='Use wandb')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    args = parser.parse_args()

    train(args)