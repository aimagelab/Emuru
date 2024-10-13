import torch
import torch.nn as nn

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
                 latent_htr_wid: bool = False):
        super().__init__()

        self.kl_weight = kl_weight
        self.alphabet = alphabet
        self.noisy_teacher = NoisyTeacherForcing(len(self.alphabet), self.alphabet.num_extra_tokens, noisy_teach_prob)

        self.latent_htr_wid = latent_htr_wid
        self.htr_weight = htr_weight
        self.htr_criterion = SmoothCrossEntropyLoss(tgt_pad_idx=self.alphabet.pad)
        self.cer = evaluate.load('cer')

        self.writer_weight = writer_weight

        self.writer_criterion = nn.CrossEntropyLoss()
        self.accuracy = evaluate.load('accuracy')

        self.log_var = nn.Parameter(torch.ones(size=()) * logvar_init)

    def forward(self, images, z, reconstructions, posteriors, writers, text_logits_s2s,
                text_logits_s2s_length, split="train", tgt_key_padding_mask=None, source_mask=None, htr=None, writer_id=None):

        images = (images + 1) / 2
        images_rgb, images_alpha = images.split([3, 1], dim=1)
        reconstructions = (reconstructions + 1) / 2
        reconstructions_rgb, reconstructions_alpha = reconstructions.split([3, 1], dim=1)

        rec_loss = torch.abs(images.contiguous() - reconstructions.contiguous())
        nll_loss = rec_loss
        htr_loss = torch.tensor(0.0, device=images.device)
        cer = torch.tensor(0.0, device=images.device)
        writer_loss = torch.tensor(0.0, device=images.device)
        acc = torch.tensor(0.0, device=images.device)
        predicted_characters_htr = []
        predicted_authors_writer_id = []

        images_grayscaled = images_rgb * images_alpha 
        images_grayscaled = 0.2989 * images_grayscaled[:, 0] + 0.5870 * images_grayscaled[:, 1] + 0.1140 * images_grayscaled[:, 2]
        images_grayscaled = 1 - images_grayscaled.unsqueeze(1)
        reconstructions_grayscaled = reconstructions_rgb * reconstructions_alpha
        reconstructions_grayscaled = 0.2989 * reconstructions_grayscaled[:, 0] + 0.5870 * reconstructions_grayscaled[:, 1] + 0.1140 * reconstructions_grayscaled[:, 2]
        reconstructions_grayscaled = 1 - reconstructions_grayscaled.unsqueeze(1)

        if htr is not None:
            text_logits_s2s_noisy = self.noisy_teacher(text_logits_s2s, text_logits_s2s_length)
            htr_input = reconstructions_grayscaled if not self.latent_htr_wid else z
            output_htr = htr(htr_input, text_logits_s2s_noisy[:, :-1], source_mask, tgt_key_padding_mask[:, :-1])
            htr_loss = self.htr_criterion(output_htr, text_logits_s2s[:, 1:]) * self.htr_weight
            predicted_logits = torch.argmax(output_htr, dim=2)
            predicted_characters = self.alphabet.decode(predicted_logits, [self.alphabet.eos])
            correct_characters = self.alphabet.decode(text_logits_s2s[:, 1:], [self.alphabet.eos])
            cer = self.cer.compute(predictions=predicted_characters, references=correct_characters)
            nll_loss = nll_loss + htr_loss
            predicted_characters_htr.append(predicted_characters)

        if writer_id is not None:
            writer_id_input = reconstructions_grayscaled if not self.latent_htr_wid else z
            output_writer_id = writer_id(writer_id_input)
            writer_loss = self.writer_criterion(output_writer_id, writers) * self.writer_weight
            predicted_authors = torch.argmax(output_writer_id, dim=1)
            acc = self.accuracy.compute(predictions=predicted_authors.int(), references=writers.int())['accuracy']
            nll_loss = nll_loss + writer_loss
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

        return {'loss': loss, 'htr_loss': htr_loss, 'writer_loss': writer_loss}, log, wandb_media_log
