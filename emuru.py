import torch
from transformers import AutoTokenizer
from transformers import T5ForConditionalGeneration, T5Config
from custom_datasets import HFDataCollector
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from torch.nn import MSELoss, CTCLoss
from pathlib import Path
from torchvision.utils import make_grid, save_image
from PIL import Image, ImageDraw, ImageFont
from models.origami import OrigamiNet
from models.autoencoder_kl import AutoencoderKL

class Emuru(torch.nn.Module):
    def __init__(self, t5_checkpoint='google-t5/t5-small',
                 vae_checkpoint='results_vae/a912/model_0205',
                 ocr_checkpoint='files/checkpoints/Origami_bw_img/origami.pth', slices_per_query=1, channels=4):
        super(Emuru, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')  # per-character tokenizer
        self.data_collator = HFDataCollector(tokenizer=self.tokenizer)

        self.padding_token = torch.tensor([[-0.4951,  0.8021,  0.3429,  0.5622,  0.5271,  0.5756,  0.7194,  0.6150]])
        self.padding_token_threshold = 0.484982096850872

        config = T5Config.from_pretrained(t5_checkpoint)
        config.vocab_size = len(self.tokenizer)
        self.T5 = T5ForConditionalGeneration(config)
        self.T5.lm_head = torch.nn.Identity()
        self.sos = torch.nn.Embedding(1, config.d_model)
        self.query_emb = torch.nn.Linear(8 * channels * slices_per_query, config.d_model)
        self.t5_to_vae = torch.nn.Linear(config.d_model, 8 * channels * slices_per_query, bias=False)
        self.t5_to_ocr = torch.nn.Linear(config.d_model, len(self.tokenizer), bias=False)

        self.vae = AutoencoderKL.from_pretrained(vae_checkpoint, subfolder='vae')
        self.set_training(self.vae, False)

        self.ocr = OrigamiNet.from_checkpoint(ocr_checkpoint, o_classes=165, n_channels=1)
        self.set_training(self.ocr, False)
        
        self.query_rearrange = Rearrange('b c h (w q) -> b w (q c h)', q=slices_per_query)
        self.z_rearrange = Rearrange('b w (q c h) -> b c h (w q)', c=channels, q=slices_per_query)
        self.z_rearrange_eval = Rearrange('w b (q c h) -> b c h (w q)', c=channels, q=slices_per_query)

        self.mse_criterion = MSELoss()
        self.ctc_criterion = CTCLoss()
        self.trainer = None
        self.alpha = 1.0

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

        if self.alpha < 1.0:
            pred_img = self.vae.decode(pred_latent).sample
            gt_img = self.vae.decode(z).sample
            ocr_preds = self.ocr(pred_img)
            ocr_gt = self.ocr(gt_img)
            ocr_loss = self.mse_criterion(ocr_preds, ocr_gt)
        else:
            ocr_loss = torch.tensor(0.0).to(mse_loss.device)

        loss = mse_loss * self.alpha + ocr_loss * (1 - self.alpha)
        return {'loss': loss, 'mse_loss': mse_loss, 'ocr_loss': ocr_loss}, pred_latent, z

    def forward_recurrent(self, text=None, img=None, input_ids=None, attention_mask=None, length=None, noise=0):
        output = self.generate(input_ids=input_ids, max_new_tokens=96, img=img, decoder_truncate=16)

        ocr_preds = self.ocr(output)
        preds_size = torch.IntTensor([ocr_preds.size(1)] * ocr_preds.size(0)).to(ocr_preds.device)
        ocr_preds = ocr_preds.permute(1, 0, 2).log_softmax(2)
        ocr_loss = self.ctc_criterion(ocr_preds, input_ids, preds_size, length)
        mse_loss = 0

        loss = mse_loss + ocr_loss
        return {'loss': loss, 'mse_loss': mse_loss, 'ocr_loss': ocr_loss}, None, None
    

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
    
    
    def generate(self, text=None, img=None, z_sequence=None, input_ids=None, max_new_tokens=256,
                 stopping_criteria='latent', stopping_after=10, stopping_errors=1):
        assert text is not None or input_ids is not None, 'Either text or input_ids must be provided'
        assert img is not None or z_sequence is not None, 'Either img or z_sequence must be provided'

        if input_ids is None:
            input_ids = self.tokenizer(text, return_tensors='pt', padding=True).input_ids
            input_ids = input_ids.to(next(self.T5.parameters()).device)
        
        if z_sequence is None:
            _, z_sequence, _ = self._img_encode(img)
        z_sequence = [z_sequence]

        sos = repeat(self.sos.weight, '1 d -> b 1 d', b=input_ids.size(0))
        for _ in range(max_new_tokens):
            if len(z_sequence) == 0:
                decoder_inputs_embeds = sos
            else:
                decoder_inputs_embeds = self.query_emb(torch.cat(z_sequence, dim=1))
                decoder_inputs_embeds = torch.cat([sos, decoder_inputs_embeds], dim=1)
            output = self.T5(input_ids, decoder_inputs_embeds=decoder_inputs_embeds)
            vae_latent = self.t5_to_vae(output.logits[:, -1:])
            z_sequence.append(vae_latent)

            if stopping_criteria == 'latent':
                curr_z_sequence = torch.cat(z_sequence, dim=1)
                pad_token = repeat(self.padding_token, '1 d -> b 1 d', b=input_ids.size(0)).to(decoder_inputs_embeds.device)
                similarity = torch.nn.functional.cosine_similarity(curr_z_sequence, pad_token, dim=-1)
                similarity = similarity[:, -stopping_after:] > self.padding_token_threshold
                if torch.all(similarity.sum(-1) >= (stopping_after - stopping_errors)):
                    # z_sequence = [curr_z_sequence[:, :-stopping_after]]
                    z_sequence = [curr_z_sequence]
                    break
            elif stopping_criteria == 'pixel':
                raise NotImplementedError

        z_sequence = torch.cat(z_sequence, dim=1)
        img = torch.clamp(self.vae.decode(self.z_rearrange(z_sequence)).sample, -1, 1)
        return img
    
    @torch.no_grad()
    def continue_gen_test(self, gt, batch, pred=None):
        def _continue_gen(style_len):
            _, z_sequence, _ = self._img_encode(batch['img'])
            z_sequence = z_sequence[:, :style_len]
            test_img = self.generate(input_ids=batch['input_ids'], z_sequence=z_sequence, max_new_tokens=96 - style_len, stopping_criteria=None)[:16]
            test_img[:, :, :, style_len * 8] = -1  # add a black line between style and pred
            return test_img
        
        if pred is not None:
            pred = torch.clamp(self.vae.decode(pred).sample, -1, 1)
        gt = torch.clamp(self.vae.decode(gt).sample, -1, 1)
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