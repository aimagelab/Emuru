import torch
import torch.nn as nn

from models.htr import HTR
from models.writer_id import WriterID
from models.smooth_ce import SmoothCrossEntropyLoss
from models.teacher_forcing import NoisyTeacherForcing
import evaluate
from custom_datasets.alphabet import Alphabet


class AutoencoderLoss(nn.Module):
    def __init__(self,
                 logvar_init: float = 0.0,
                 kl_weight: float = 1e-06,
                 htr_weight: float = 0.3,
                 writer_weight: float = 0.005,
                 noisy_teach_prob: float = 0.3,
                 alphabet: Alphabet = None,
                 # htr: HTR = None,
                 #writer_id: WriterID = None,
                 # latent_htr_wid: bool = False):
                 htr_path: str = None,
                 htr_config: str = None,
                 writer_id_path: str = None,
                 writer_id_config: str = None,
                 latent_htr_wid: bool = False):
        super().__init__()

        self.kl_weight = kl_weight
        self.alphabet = alphabet
        self.noisy_teacher = NoisyTeacherForcing(len(self.alphabet), self.alphabet.num_extra_tokens, noisy_teach_prob)

        self.latent_htr_wid = latent_htr_wid
        self.htr_weight = htr_weight
        self.htr_criterion = SmoothCrossEntropyLoss(tgt_pad_idx=self.alphabet.pad)
        self.cer = evaluate.load('cer')
        self.htr = None

        self.writer_weight = writer_weight
        self.writer_id = None
        
        if self.latent_htr_wid:
            if htr_config is not None: 
                self.htr = HTR.from_config(htr_config)
                self.htr.train()
            if writer_id_config is not None:
                self.writer_id = WriterID.from_config(writer_id_config)
                self.writer_id.train()
        else:
            if htr_path is not None and htr_weight > 0:
                self.htr = HTR.from_pretrained(htr_path)
                self.htr.eval()
                for param in self.htr.parameters():
                    param.requires_grad = False
            
            if writer_id_path is not None and self.writer_weight > 0:
                self.writer_id = WriterID.from_pretrained(writer_id_path)
                self.writer_id.eval()
                for param in self.writer_id.parameters():
                    param.requires_grad = False

        self.writer_criterion = nn.CrossEntropyLoss()
        self.accuracy = evaluate.load('accuracy')

        self.log_var = nn.Parameter(torch.ones(size=()) * logvar_init)

    def forward(self, images, z, reconstructions, posteriors, writers, text_logits_s2s,
                text_logits_s2s_length, split="train", tgt_key_padding_mask=None, source_mask=None):

        rec_loss = torch.abs(images.contiguous() - reconstructions.contiguous())
        nll_loss = rec_loss
        htr_loss = torch.tensor(0.0, device=images.device)
        cer = torch.tensor(0.0, device=images.device)
        writer_loss = torch.tensor(0.0, device=images.device)
        acc = torch.tensor(0.0, device=images.device)
        predicted_characters_htr = []
        predicted_authors_writer_id = []

        if self.htr is not None:
            text_logits_s2s_noisy = self.noisy_teacher(text_logits_s2s, text_logits_s2s_length)
            htr_input = reconstructions if not self.latent_htr_wid else z
            output_htr = self.htr(htr_input, text_logits_s2s_noisy[:, :-1], source_mask, tgt_key_padding_mask[:, :-1])
            htr_loss = self.htr_criterion(output_htr, text_logits_s2s[:, 1:])
            predicted_logits = torch.argmax(output_htr, dim=2)
            predicted_characters = self.alphabet.decode(predicted_logits, [self.alphabet.eos])
            correct_characters = self.alphabet.decode(text_logits_s2s[:, 1:], [self.alphabet.eos])
            cer = self.cer.compute(predictions=predicted_characters, references=correct_characters)
            nll_loss = nll_loss + self.htr_weight * htr_loss
            predicted_characters_htr.append(predicted_characters)

        if self.writer_id is not None:
            writer_id_input = reconstructions if not self.latent_htr_wid else z
            output_writer_id = self.writer_id(writer_id_input)
            writer_loss = self.writer_criterion(output_writer_id, writers)
            predicted_authors = torch.argmax(output_writer_id, dim=1)
            acc = self.accuracy.compute(predictions=predicted_authors.int(), references=writers.int())['accuracy']
            nll_loss = nll_loss + self.writer_weight * writer_loss
            predicted_authors_writer_id.append(list(predicted_authors))

        nll_loss = nll_loss / torch.exp(self.log_var) + self.log_var
        nll_loss = nll_loss.mean()
        kl_loss = posteriors.kl().mean()

        loss = nll_loss + self.kl_weight * kl_loss

        log = {f"{split}/total_loss": loss.detach().mean().item(),
               f"{split}/log_var": self.log_var.detach().item(),
               f"{split}/kl_loss": kl_loss.detach().mean().item(),
               f"{split}/nll_loss": nll_loss.detach().mean().item(),
               f"{split}/rec_loss": rec_loss.detach().mean().item(),
               f"{split}/writer_loss": writer_loss.detach().mean().item(),
               f"{split}/HTR_loss": htr_loss.detach().mean().item(),
               f"{split}/cer": cer,
               f"{split}/acc": acc,
               }

        wandb_media_log = {
            f'{split}/predicted_characters': predicted_characters_htr,
            f'{split}/predicted_authors': predicted_authors_writer_id
        }

        return loss, log, wandb_media_log
