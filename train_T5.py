import torch
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import T5ForConditionalGeneration, T5Config
from diffusers import AutoencoderKL
from custom_datasets import OnlineFontSquare, HFDataCollector, TextSampler, FixedTextSampler
from einops.layers.torch import Rearrange
from torch.nn import MSELoss
from pathlib import Path
from torch.utils.data import DataLoader
import wandb
import argparse
from tqdm import tqdm
from utils import MetricCollector
from torchvision.utils import make_grid


class HTGT5(torch.nn.Module):
    def __init__(self, checkpoint="google-t5/t5-small", vae_checkpoint='files/checkpoints/ca5f'):
        super(HTGT5, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.data_collator = HFDataCollector(tokenizer=self.tokenizer)

        config = T5Config.from_pretrained(checkpoint)
        self.model = T5ForConditionalGeneration(config)
        self.model.lm_head = torch.nn.Linear(config.d_model, 8, bias=False)

        self.vae = AutoencoderKL.from_pretrained(vae_checkpoint)
        self.set_vae_training(False)
        
        self.query_emb = torch.nn.Linear(8, config.d_model)
        self.query_rearrange = Rearrange('b c h w -> b w (c h)')
        self.z_rearrange = Rearrange('b w (c h) -> b c h w', c=1)

        self.criterion = MSELoss()
        self.trainer = None

    def set_vae_training(self, training):
        self.vae.train() if training else self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = training

    def forward(self, text=None, img=None, input_ids=None, attention_mask=None, length=None):
        posterior = self.vae.encode(img.float()).latent_dist
        z = posterior.sample()
        z_sequence = self.query_rearrange(z)
        decoder_inputs_embeds = self.query_emb(z_sequence)
        output = self.model(input_ids, attention_mask=attention_mask, decoder_inputs_embeds=decoder_inputs_embeds)
        loss = self.criterion(output.logits, z_sequence)
        return loss, self.z_rearrange(output.logits), z


def train(args):
    model = HTGT5(args.checkpoint, args.vae_checkpoint).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    dataset = OnlineFontSquare('files/font_square/fonts', 'files/font_square/backgrounds', TextSampler(8, 32, 4))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=model.data_collator,
                        num_workers=args.dataloader_num_workers, persistent_workers=args.dataloader_num_workers > 0)

    eval_fonts = sorted(Path('files/font_square/fonts').rglob('*.ttf'))[:100]
    dataset_eval = OnlineFontSquare(eval_fonts, [], FixedTextSampler('this is a test'))
    loader_eval = DataLoader(dataset_eval, batch_size=args.batch_size, shuffle=False, collate_fn=model.data_collator, num_workers=args.dataloader_num_workers)

    wandb.init(project='htgt5', config=args)
    collector = MetricCollector()

    for epoch in range(args.num_train_epochs):
        model.train()
        for i, batch in tqdm(enumerate(loader), total=len(loader), desc=f'Epoch {epoch}'):
            batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            loss, pred, gt = model(**batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            collector['mse_loss'] = loss.item()

        # model.eval()
        # for i, batch in tqdm(enumerate(loader_eval), total=len(loader_eval), desc=f'Evaluating'):
        #     batch = {k: v.to(args.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        #     res = model(**batch)

        with torch.no_grad():
            model.eval()
            pred = model.vae.decoder(pred).repeat(1, 3, 1, 1)
            gt = model.vae.decoder(gt).repeat(1, 3, 1, 1)
            img = torch.cat([batch['img'], pred, gt], dim=-1)[:16]
            wandb.log({
                'img_recon_pred': wandb.Image(make_grid(img, nrow=1, normalize=True)),
            } | collector.dict())

# training_args = Seq2SeqTrainingArguments(
#     output_dir="files/checkpoints",
#     evaluation_strategy="epoch",
#     learning_rate=2e-5,
#     per_device_train_batch_size=64,
#     per_device_eval_batch_size=64,
#     weight_decay=0.01,
#     save_total_limit=3,
#     num_train_epochs=100,
#     predict_with_generate=True,
#     fp16=False,
#     push_to_hub=False,
#     report_to='wandb',  # 'wandb',
#     dataloader_num_workers=8
# )



# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=dataset,
#     eval_dataset=dataset_eval,
#     tokenizer=model.tokenizer,
#     data_collator=model.data_collator,
#     compute_metrics=None,
# )

# model.trainer = trainer

# trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a T5 model with a VAE')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--checkpoint', type=str, default='google-t5/t5-small', help='T5 checkpoint')
    parser.add_argument('--vae_checkpoint', type=str, default='files/checkpoints/ca5f', help='VAE checkpoint')
    parser.add_argument('--output_dir', type=str, default='files/checkpoints', help='Output directory')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--num_train_epochs', type=int, default=100, help='Number of train epochs')
    parser.add_argument('--report_to', type=str, default='none', help='Report to')
    parser.add_argument('--dataloader_num_workers', type=int, default=8, help='Dataloader num workers')
    args = parser.parse_args()

    train(args)