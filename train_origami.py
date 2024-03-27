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


class Emuru(torch.nn.Module):
    def __init__(self, t5_checkpoint="google-t5/t5-small", vae_checkpoint='files/checkpoints/lightning_vae/vae.ckpt', slices_per_query=1):
        super(Emuru, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')  # per-character tokenizer
        self.data_collator = HFDataCollector(tokenizer=self.tokenizer)

        config = T5Config.from_pretrained(t5_checkpoint)
        config.vocab_size = len(self.tokenizer)
        self.T5 = T5ForConditionalGeneration(config)
        self.T5.lm_head = torch.nn.Linear(config.d_model, 8 * slices_per_query, bias=False)

        self.sos = torch.nn.Embedding(1, config.d_model)

        # self.vae = AutoencoderKL.from_pretrained(vae_checkpoint)
        self.vae = LightningAutoencoderKL.load_from_checkpoint(vae_checkpoint, strict=False)
        self.set_vae_training(False)
        
        self.query_emb = torch.nn.Linear(8 * slices_per_query, config.d_model)
        self.query_rearrange = Rearrange('b c h (w q) -> b w (q c h)', q=slices_per_query)
        self.z_rearrange = Rearrange('b w (q c h) -> b c h (w q)', c=1, q=slices_per_query)
        self.z_rearrange_eval = Rearrange('w b (q c h) -> b c h (w q)', c=1, q=slices_per_query)

        self.criterion = MSELoss()
        self.trainer = None

    def set_vae_training(self, training):
        self.vae.train() if training else self.vae.eval()
        for param in self.vae.parameters():
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
        loss = self.criterion(output.logits[:, :-1], z_sequence)
        return loss, self.z_rearrange(output.logits[:, :-1]), z
    
    def generate(self, text=None, img=None, input_ids=None, max_new_tokens=96, decoder_truncate=None):
        assert text is not None or input_ids is not None, 'Either text or input_ids must be provided'
        if input_ids is None:
            input_ids = self.tokenizer(text, return_tensors='pt').input_ids
            input_ids = input_ids.to(next(self.T5.parameters()).device)

        decoder_inputs_embeds, z_sequence, z = None, None, None
        if img is not None:
            decoder_inputs_embeds, z_sequence, z = self._img_encode(img)

        if decoder_truncate is not None and z_sequence is not None:
            decoder_inputs_embeds = decoder_inputs_embeds[:, :decoder_truncate]
            z_sequence = z_sequence[:, :decoder_truncate]
            z = self.z_rearrange(z_sequence)
        
        output = self.T5.generate(input_ids,
                                  max_new_tokens=max_new_tokens,
                                  return_dict_in_generate=True,
                                  output_scores=True,
                                  decoder_inputs_embeds=decoder_inputs_embeds,
                                  )
        
        scores = self.z_rearrange_eval(torch.stack(output.scores))
        if z is not None:
            scores = torch.cat([z, scores], dim=-1)
        img = torch.clamp(self.vae.decode(scores), -1, 1)
        return img
    
    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.T5.state_dict(), path / 'T5.pth')
        torch.save(self.vae.state_dict(), path / 'VAE.pth')
        torch.save(self.query_emb.state_dict(), path / 'query_emb.pth')
        torch.save(self.sos.state_dict(), path / 'sos.pth')

    def load_pretrained(self, path):
        path = Path(path)
        self.T5.load_state_dict(torch.load(path / 'T5.pth'))
        self.vae.load_state_dict(torch.load(path / 'VAE.pth'))
        self.query_emb.load_state_dict(torch.load(path / 'query_emb.pth'))
        self.sos.load_state_dict(torch.load(path / 'sos.pth'))


def train(args):
    text_sampler = TextSampler(8, 32, 4)
    dataset = OnlineFontSquare('files/font_square/fonts', 'files/font_square/backgrounds', text_sampler)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn, drop_last=True,
                        num_workers=args.dataloader_num_workers, persistent_workers=args.dataloader_num_workers > 0)

    model = OrigamiNet(len(text_sampler.charset) + 1, n_channels=1).to(args.device)
    text_encoder = CTCLabelConverter(text_sampler.charset).to(args.device)
    vae = LightningAutoencoderKL.load_from_checkpoint(args.vae_checkpoint, strict=False).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = torch.nn.CTCLoss()

    # eval_fonts = sorted(Path('files/font_square/fonts').rglob('*.ttf'))[:100]
    # dataset_eval = OnlineFontSquare(eval_fonts, [], FixedTextSampler('this is a test'))
    # loader_eval = DataLoader(dataset_eval, batch_size=args.batch_size, shuffle=False, collate_fn=model.data_collator, num_workers=args.dataloader_num_workers)

    if args.wandb: wandb.init(project='Emuru_origami', config=args)
    collector = MetricCollector()

    for epoch in range(args.num_train_epochs):
        model.train()
        for i, batch in tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch}'):
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            enc_text, enc_text_len = text_encoder.encode(batch['text'], args.device)

            with torch.no_grad():
                posterior = vae.encode(batch['bw_img'].float())
                z = posterior.sample()
            preds = model(z)

            preds_size = torch.IntTensor([preds.size(1)] * args.batch_size).to(args.device)
            preds = preds.permute(1, 0, 2).log_softmax(2)
            loss = criterion(preds, enc_text, preds_size, enc_text_len)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            collector['ctc_loss'] = loss.item()

        with torch.no_grad():
            model.eval()
            pred = torch.clamp(model.vae.decode(pred), -1, 1)
            gt = torch.clamp(model.vae.decode(gt), -1, 1)
            img = torch.cat([batch['img'], gt, pred], dim=-1)[:16]
            test_img = model.generate(input_ids=batch['input_ids'], img=None, max_new_tokens=96 - 16, decoder_truncate=16)[:16]
            if args.wandb: wandb.log({
                # 'img_recon_pred': wandb.Image(make_grid(img, nrow=1, normalize=True)),
                # 'this_is_a_test': wandb.Image(make_grid(test_img, nrow=1, normalize=True)),
            } | collector.dict())
                
        # if epoch % 10 == 0 and epoch > 0:
        #     model.save_pretrained(args.output_dir)
        #     print(f'Saved model at epoch {epoch} in {args.output_dir}')
        
        collector.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a T5 model with a VAE')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--t5_checkpoint', type=str, default='google-t5/t5-small', help='T5 checkpoint')
    parser.add_argument('--vae_checkpoint', type=str, default='files/checkpoints/lightning_vae/vae.ckpt', help='VAE checkpoint')
    parser.add_argument('--output_dir', type=str, default='files/checkpoints/Emuru_sm', help='Output directory')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--num_train_epochs', type=int, default=10 ** 10, help='Number of train epochs')
    parser.add_argument('--report_to', type=str, default='none', help='Report to')
    parser.add_argument('--dataloader_num_workers', type=int, default=12, help='Dataloader num workers')
    parser.add_argument('--slices_per_query', type=int, default=1, help='Number of slices to predict in each query')
    parser.add_argument('--wandb', action='store_true', help='Use wandb')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    args = parser.parse_args()

    train(args)