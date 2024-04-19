import torch
import copy
from transformers import AutoTokenizer
from transformers import T5ForConditionalGeneration, T5Config
from custom_datasets import OnlineFontSquare, HFDataCollector, TextSampler, FixedTextSampler, dataset_factory
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from torch.nn import MSELoss, CTCLoss
from pathlib import Path
from torch.utils.data import DataLoader
import wandb
import argparse
from tqdm import tqdm
from utils import MetricCollector
from torchvision.utils import make_grid, save_image
from PIL import Image, ImageDraw, ImageFont

from models.autoencoder_kl import AutoencoderKL
# from models import OrigamiNet

class Emuru(torch.nn.Module):
    def __init__(self,
                 t5_checkpoint="google-t5/t5-small",
                 vae_checkpoint='files/checkpoints/vae_0850',
                 ocr_checkpoint='files/checkpoints/Origami_bw_img/origami.pth',
                 slices_per_query=1):
        super(Emuru, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')  # per-character tokenizer
        self.data_collator = HFDataCollector(tokenizer=self.tokenizer)

        config = T5Config.from_pretrained(t5_checkpoint)
        config.vocab_size = len(self.tokenizer)
        self.T5 = T5ForConditionalGeneration(config)
        self.T5.lm_head = torch.nn.Identity()
        self.sos = torch.nn.Embedding(1, config.d_model)
        self.query_emb = torch.nn.Linear(8 * slices_per_query, config.d_model)
        self.t5_to_vae = torch.nn.Linear(config.d_model, 8 * slices_per_query, bias=False)
        self.t5_to_ocr = torch.nn.Linear(config.d_model, len(self.tokenizer), bias=False)

        self.vae = AutoencoderKL.from_pretrained(vae_checkpoint)
        self.set_training(self.vae, False)

        # self.ocr = OrigamiNet.from_checkpoint(ocr_checkpoint, o_classes=165, n_channels=1)
        # self.set_training(self.ocr, False)
        
        self.query_rearrange = Rearrange('b c h (w q) -> b w (q c h)', q=slices_per_query)
        self.z_rearrange = Rearrange('b w (q c h) -> b c h (w q)', c=1, q=slices_per_query)
        self.z_rearrange_eval = Rearrange('w b (q c h) -> b c h (w q)', c=1, q=slices_per_query)

        self.mse_criterion = MSELoss()
        self.ctc_criterion = CTCLoss()
        self.trainer = None

    def set_training(self, model, training):
        model.train() if training else model.eval()
        for param in model.parameters():
            param.requires_grad = training 

    def _img_encode(self, img, noise=0):
        posterior = self.vae.encode(img.float())
        z = posterior.latent_dist.sample()
        z_sequence = self.query_rearrange(z)

        noise_sequence = z_sequence
        if noise > 0:
            noise_sequence = z_sequence + torch.randn_like(z_sequence) * noise

        decoder_inputs_embeds = self.query_emb(noise_sequence)
        sos = repeat(self.sos.weight, '1 d -> b 1 d', b=decoder_inputs_embeds.size(0))
        decoder_inputs_embeds = torch.cat([sos, decoder_inputs_embeds], dim=1)
        return decoder_inputs_embeds, z_sequence, z

    def forward(self, text=None, img=None, input_ids=None, attention_mask=None, length=None, noise=0):
        decoder_inputs_embeds, z_sequence, z = self._img_encode(img, noise)

        output = self.T5(input_ids, attention_mask=attention_mask, decoder_inputs_embeds=decoder_inputs_embeds)
        vae_latent = self.t5_to_vae(output.logits[:, :-1])
        pred_latent = self.z_rearrange(vae_latent)

        mse_loss = self.mse_criterion(vae_latent, z_sequence)

        # ocr_preds = self.t5_to_ocr(output.logits)
        # preds_size = torch.IntTensor([ocr_preds.size(1)] * ocr_preds.size(0)).to(ocr_preds.device)
        # ocr_preds = ocr_preds.permute(1, 0, 2).log_softmax(2)
        # ocr_loss = self.ctc_criterion(ocr_preds, input_ids, preds_size, length)
        ocr_loss = 0

        loss = mse_loss + ocr_loss

        # idx = 0
        # self.split_characters(pred_latent[idx:idx+1], z[idx:idx+1], ocr_preds.argmax(-1)[:, idx])

        return {'loss': loss, 'mse_loss': mse_loss, 'ocr_loss': ocr_loss}, pred_latent, z
    

    def split_characters(self, pred, gt, indices):
        pred = self.vae.decode(pred).sample
        gt = self.vae.decode(gt).sample
        img = torch.cat([gt, pred], dim=-2)

        curr_char = indices[0]
        for idx, char in enumerate(indices):
            if char != curr_char:
                img[:, :, :, idx * 8 - 1] = -1
                curr_char = char

        img = self.write_text_below_image(img, self.tokenizer.decode(indices))

        return img
    

    @torch.no_grad()
    def write_text_below_image(self, image, text):
        image = (torch.clamp(image, -1, 1) + 1) * 127.5
        image = rearrange(image.to(torch.uint8), '1 1 h w -> h w').cpu().numpy()
        image = Image.fromarray(image, mode='L')

        text = text.replace('<pad>', '#').replace('</s>', '$')

        # Load the font
        font = ImageFont.load_default()
        ascent, descent = font.getmetrics()
        (width, baseline), (offset_x, offset_y) = font.font.getsize(text)

        # Calculate dimensions for the new image
        img_width, img_height = image.size
        new_height = img_height + offset_y + ascent +descent

        # Create a new image with white background
        new_image = Image.new('L', (img_width, new_height), color='white')

        # Paste the original image onto the new image
        new_image.paste(image, (0, 0))

        # Draw the text onto the new image
        draw = ImageDraw.Draw(new_image)

        curr_char = None
        for idx, char in enumerate(text):
            if char != curr_char:
                curr_char = char
                draw.text((idx * 8, img_height), char, fill='black', font=font)

        return new_image
    
    
    @torch.no_grad()
    def generate(self, text=None, img=None, input_ids=None, max_new_tokens=96, decoder_truncate=None):
        assert text is not None or input_ids is not None, 'Either text or input_ids must be provided'
        if input_ids is None:
            input_ids = self.tokenizer(text, return_tensors='pt', padding=True).input_ids
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
            vae_latent = self.t5_to_vae(output.logits[:, -1:])
            new_z_sequence.append(vae_latent)

        z_sequence = torch.cat(new_z_sequence, dim=1)
        img = torch.clamp(self.vae.decode(self.z_rearrange(z_sequence)).sample, -1, 1)
        return img
    
    def continue_gen_test(self, pred, gt, batch):
        def _continue_gen(style_len):
            test_img = self.generate(input_ids=batch['input_ids'], img=batch['img'], max_new_tokens=96 - style_len, decoder_truncate=style_len)[:16]
            test_img[:, :, :, style_len * 8] = -1  # add a black line between style and pred
            return test_img
        
        pred = repeat(torch.clamp(self.vae.decode(pred).sample, -1, 1), 'b 1 h w -> b 3 h w')
        gt = repeat(torch.clamp(self.vae.decode(gt).sample, -1, 1), 'b 1 h w -> b 3 h w')
        return pred, gt, torch.cat([make_grid(gt[:16], nrow=1, normalize=True),
                            make_grid(_continue_gen(1), nrow=1, normalize=True),
                            make_grid(_continue_gen(4), nrow=1, normalize=True),
                            make_grid(_continue_gen(8), nrow=1, normalize=True),
                            make_grid(_continue_gen(16), nrow=1, normalize=True),
                            make_grid(_continue_gen(32), nrow=1, normalize=True),
                            ], dim=-1)
    
    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.T5.state_dict(), path / 'T5.pth')
        torch.save(self.vae.state_dict(), path / 'VAE.pth')
        # torch.save(self.ocr.state_dict(), path / 'OCR.pth')
        torch.save(self.query_emb.state_dict(), path / 'query_emb.pth')
        torch.save(self.sos.state_dict(), path / 'sos.pth')

    def load_pretrained(self, path):
        path = Path(path)
        self.T5.load_state_dict(torch.load(path / 'T5.pth'))
        self.vae.load_state_dict(torch.load(path / 'VAE.pth'))
        # self.ocr.load_state_dict(torch.load(path / 'OCR.pth'))
        self.query_emb.load_state_dict(torch.load(path / 'query_emb.pth'))
        self.sos.load_state_dict(torch.load(path / 'sos.pth'))


def train(args):
    if args.device == 'cpu':
        print('WARNING: Using CPU')

    model = Emuru(args.t5_checkpoint, args.vae_checkpoint, args.ocr_checkpoint, args.slices_per_query).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    if args.resume:
        try:
            checkpoint_path = sorted(Path(args.output_dir).rglob('*.pth'))[-1]
            checkpoint = torch.load(checkpoint_path, map_location=args.device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            args.wandb_id = checkpoint['wandb_id']
            print(f'Resumed training from {args.output_dir}')
        except KeyError:
            model.load_pretrained(args.output_dir)
            print(f'Resumed with the old checkpoint system: {args.output_dir}')

    dataset = OnlineFontSquare('files/font_square/fonts', 'files/font_square/backgrounds', TextSampler(8, 32, (4, 7)))
    dataset.length *= args.db_multiplier
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=model.data_collator,
                        num_workers=args.dataloader_num_workers, persistent_workers=args.dataloader_num_workers > 0)
    
    eval_dataset = dataset_factory('test', ['iam_lines'], root_path='/home/vpippi/Teddy/files/datasets/')
    eval_dataset.batch_keys('style')
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=eval_dataset.collate_fn,
                            num_workers=args.dataloader_num_workers_eval, persistent_workers=False)

    # eval_fonts = sorted(Path('files/font_square/fonts').rglob('*.ttf'))[:100]
    # dataset_eval = OnlineFontSquare(eval_fonts, [], FixedTextSampler('this is a test'))
    # loader_eval = DataLoader(dataset_eval, batch_size=args.batch_size, shuffle=False, collate_fn=model.data_collator, num_workers=args.dataloader_num_workers)

    if args.wandb:
        resume = 'must' if args.resume else 'allow'
        wandb.init(project='Emuru', config=args, id=args.wandb_id, resume=resume)
    collector = MetricCollector()

    for epoch in range(args.start_epoch, args.num_train_epochs):
        model.train()
        for i, batch in tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch}'):
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            batch['noise'] = args.teacher_noise

            losses, pred, gt = model(**batch)
            optimizer.zero_grad()
            losses['loss'].backward()
            optimizer.step()

            losses = {f'train/{k}': v for k, v in losses.items()}
            collector.update(losses)

            # print('Warning')
            # if i > 2:
            #     break
            # imgs = model.custom_generate(text='this is a sample text', img=None, max_new_tokens=96)
            # imgs = model.custom_generate(input_ids=batch['input_ids'], img=batch['img'], max_new_tokens=96 - 16, decoder_truncate=16)
            # print()

        with torch.no_grad():
            model.eval()
            wandb_data = {}

            pred, gt, synth_gen_test = model.continue_gen_test(pred, gt, batch)
            synth_img = torch.cat([batch['img'], gt, pred], dim=-1)[:16]
            wandb_data['synth_img'] = wandb.Image(make_grid(synth_img, nrow=1, normalize=True))
            wandb_data['synth_gen_test'] = wandb.Image(synth_gen_test)
            
            for i, batch in tqdm(enumerate(eval_loader), total=len(eval_loader), desc=f'Eval'):
                batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                res = model.tokenizer(batch['style_text'], return_tensors='pt', padding=True, return_attention_mask=True, return_length=True)
                res = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in res.items()}

                losses, pred, gt = model(img=batch['style_img'], **res)
                losses = {f'eval/{k}': v for k, v in losses.items()}
                collector.update(losses)

            batch['input_ids'] = model.tokenizer(batch['style_text'], return_tensors='pt', padding=True).input_ids.to(args.device)
            batch['img'] = batch['style_img']
            pred, gt, real_gen_test = model.continue_gen_test(pred, gt, batch)
            real_img = torch.cat([batch['img'], gt, pred], dim=-1)[:16]
            wandb_data['real_img'] = wandb.Image(make_grid(real_img, nrow=1, normalize=True))
            wandb_data['real_gen_test'] = wandb.Image(real_gen_test)

            if args.wandb: wandb.log(wandb_data | collector.dict())
                
        if epoch % 5 == 0 and epoch > 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'wandb_id': args.wandb_id
            }
            checkpoint_path = Path(args.output_dir) / f'{epoch // 100 * 100:05d}.pth'
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, checkpoint_path)
            print(f'Saved model at epoch {epoch} in {checkpoint_path}')

        collector.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a T5 model with a VAE')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--t5_checkpoint', type=str, default='google-t5/t5-small', help='T5 checkpoint')
    parser.add_argument('--vae_checkpoint', type=str, default='files/checkpoints/vae_0850', help='VAE checkpoint')
    parser.add_argument('--ocr_checkpoint', type=str, default='files/checkpoints/Origami_bw_img/origami.pth', help='OCR checkpoint')
    parser.add_argument('--output_dir', type=str, default='files/checkpoints/Emuru_sm', help='Output directory')
    parser.add_argument('--db_multiplier', type=int, default=10, help='Dataset multiplier')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--num_train_epochs', type=int, default=10 ** 10, help='Number of train epochs')
    parser.add_argument('--report_to', type=str, default='none', help='Report to')
    parser.add_argument('--dataloader_num_workers', type=int, default=10, help='Dataloader num workers')
    parser.add_argument('--dataloader_num_workers_eval', type=int, default=4, help='Dataloader num workers')
    parser.add_argument('--slices_per_query', type=int, default=1, help='Number of slices to predict in each query')
    parser.add_argument('--wandb', action='store_true', help='Use wandb')
    parser.add_argument('--wandb_id', type=str, default=wandb.util.generate_id(), help='Wandb id')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    parser.add_argument('--start_epoch', type=int, default=0, help='Start epoch')
    parser.add_argument('--teacher_noise', type=float, default=0.1, help='Start epoch')
    args = parser.parse_args()

    train(args)