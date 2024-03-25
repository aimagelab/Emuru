import torch
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import T5ForConditionalGeneration, T5Config
from diffusers import AutoencoderKL
from custom_datasets import OnlineFontSquare, HFDataCollector, TextSampler, FixedTextSampler
from einops.layers.torch import Rearrange
from torch.nn import MSELoss
from pathlib import Path
import wandb


class HTGT5(torch.nn.Module):
    def __init__(self, checkpoint="google-t5/t5-small", vae_checkpoint='files/checkpoints/ca5f'):
        super(HTGT5, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.data_collator = HFDataCollector(tokenizer=self.tokenizer)

        config = T5Config.from_pretrained(checkpoint)
        self.model = T5ForConditionalGeneration(config)
        self.model.lm_head = torch.nn.Linear(config.d_model, 8, bias=False)

        self.vae = AutoencoderKL.from_pretrained(vae_checkpoint)
        self.vae.eval()

        for param in self.vae.parameters():
            param.requires_grad = False
        
        self.query_emb = torch.nn.Linear(8, config.d_model)
        self.query_rearrange = Rearrange('b c h w -> b w (c h)')

        self.criterion = MSELoss()
        self.trainer = None

    def forward(self, text=None, img=None, input_ids=None, attention_mask=None, length=None):
        posterior = self.vae.encode(img.float()).latent_dist
        z_sequence = self.query_rearrange(posterior.sample())
        decoder_inputs_embeds = self.query_emb(z_sequence)
        output = self.model(input_ids, attention_mask=attention_mask, decoder_inputs_embeds=decoder_inputs_embeds)
        loss = self.criterion(output.logits, z_sequence)
        wandb.log({'train/mse_loss': loss}, step=self.trainer.state.global_step)
        return {'loss': loss}

# data = tokenizer('this is a random test')
# input_ids = torch.tensor(data['input_ids']).unsqueeze(0)
# attention_mask = torch.tensor(data['attention_mask']).unsqueeze(0)
# decoder_inputs_embeds = torch.randn(1, 64, 512)  # b l d
# res = model(input_ids, attention_mask=attention_mask, decoder_inputs_embeds=decoder_inputs_embeds)
        
model = HTGT5()

training_args = Seq2SeqTrainingArguments(
    output_dir="files/checkpoints",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=100,
    predict_with_generate=True,
    fp16=False,
    push_to_hub=False,
    report_to='wandb',  # 'wandb',
    dataloader_num_workers=8
)

dataset = OnlineFontSquare('files/font_square/fonts', 'files/font_square/backgrounds', TextSampler(8, 32, 4))

eval_fonts = sorted(Path('files/font_square/fonts').rglob('*.ttf'))[:100]
dataset_eval = OnlineFontSquare(eval_fonts, [], FixedTextSampler('this is a test'))

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset_eval,
    tokenizer=model.tokenizer,
    data_collator=model.data_collator,
    compute_metrics=None,
)

model.trainer = trainer

trainer.train()