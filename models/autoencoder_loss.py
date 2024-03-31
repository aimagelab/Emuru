import torch
import torch.nn as nn

from thirdparty.VQVAEGAN.TamingTransformers.contperceptual import *
import Parameters as pa
#   taming.modules.losses.vqperceptual import *  # TODO: taming dependency yes/no?
from src.data.augmentation.Noisy_teacher_forcing import NoisyTeacherForcing
from src.Losses.SmootheCE import SmoothCE
from src.data.utils.constants import *
from src.model.modules.HTR_Writer import HTR_Writer
from src.model.modules.WriterSequence import WriterSequence
import torchmetrics
from src.data.utils.alphabet import Alphabet
from src.utils.utils import *

from models.htr import HTR
from models.writer_id import WriterID
from smooth_ce import SmoothCrossEntropyLoss
from discriminator import NLayerDiscriminator, weights_init, hinge_d_loss, vanilla_d_loss
import evaluate


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


class AutoencoderLoss(nn.Module):
    def __init__(self,
                 disc_start: int = 9980001,
                 logvar_init: float = 0.0,
                 kl_weight: float = 1e-06,
                 pixelloss_weight: float = 1.0,
                 disc_num_layers: int = 3,
                 disc_in_channels: int = 1,
                 disc_factor: float = 1.0,
                 disc_weight: float = 0.0,
                 htr_weight: float = 0.3,
                 writer_weight: float = 0.005,
                 use_actnorm: bool = False,
                 disc_conditional: bool = False,
                 disc_loss: str = "hinge",
                 noisy_teach_prob: float = 0.3,
                 alphabet: Alphabet = None,
                 htr_config: dict = None,
                 writer_config: dict = None):
        super().__init__()

        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.alphabet = alphabet
        self.noisy_teacher = NoisyTeacherForcing(len(self.alphabet), noisy_teach_prob)

        self.writer_weight = writer_weight
        if htr_config is not None and htr_weight > 0:
            self.htr = HTR.from_config(htr_config)
            self.htr.eval()
            self.htr.freeze()
        else:
            self.htr = None

        if writer_config is not None and writer_weight > 0:
            self.writer = WriterID.from_config(writer_config)
            self.writer.eval()
            self.writer.freeze()
        else:
            self.writer = None

        self.writer_criterion = nn.CrossEntropyLoss()
        self.accuracy = evaluate.load('accuracy')
        self.htr_criterion = SmoothCrossEntropyLoss(tgt_pad_idx=self.alphabet.pad)
        self.cer = evaluate.load('cer')

        self.log_var = nn.Parameter(torch.ones(size=()) * logvar_init)
        self.discriminator = NLayerDiscriminator(
            input_nc=disc_in_channels, n_layers=disc_num_layers, use_actnorm=use_actnorm).apply(weights_init)

        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def forward(self, images, reconstructions, posteriors, global_step, writers, text_logits_s2s,
                text_logits_s2s_lenght, last_layer=None, split="train", tgt_key_padding_mask=None, source_mask=None,
                optimizer_idx=None):

        rec_loss = torch.abs(images.contiguous() - reconstructions.contiguous())
        nll_loss = rec_loss
        writer_loss = torch.tensor(0.0, device=images.device)
        htr_loss = torch.tensor(0.0, device=images.device)
        htr_rec_loss = torch.tensor(0.0, device=images.device)
        cer = torch.tensor(0.0, device=images.device)
        acc = torch.tensor(0.0, device=images.device)

        if self.htr is not None:
            text_logits_s2s_noisy = self.noisy_teacher(text_logits_s2s, text_logits_s2s_lenght)
            output_htr = self.htr(images, text_logits_s2s_noisy[:, :-1], source_mask, tgt_key_padding_mask[:, :-1])
            htr_loss = self.htr_criterion(output_htr, text_logits_s2s[:, 1:])
            predicted_logits = torch.argmax(output_htr, dim=2)
            predicted_characters = self.alphabet.decode(predicted_logits, [self.alphabet.eos])
            correct_characters = self.alphabet.decode(text_logits_s2s[:, 1:], [self.alphabet.eos])
            cer = self.cer.compute(predictions=predicted_characters, references=correct_characters)
            nll_loss = nll_loss + self.htr_weight * htr_loss

        if self.writer is not None:
            output_writer_id = self.writer(images)
            writer_loss = self.writer_criterion(output_writer_id, writers)
            predicted_authors = torch.argmax(output_writer_id, dim=1)
            accuracy_value = self.accuracy.compute(predictions=predicted_authors.int(), references=writers.int())[
                'accuracy']
            nll_loss = nll_loss + self.writer_weight * writer_loss

        nll_loss = nll_loss / torch.exp(self.logvar) + self.logvar
        nll_loss = nll_loss.mean()
        kl_loss = posteriors.kl().mean()

        if optimizer_idx == 0:
            # GAN generator update
            logits_fake = self.discriminator(reconstructions.contiguous())
            g_loss = -torch.mean(logits_fake)
            d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {f"{split}/total_loss": loss.clone().detach().mean(),
                   f"{split}/logvar": self.logvar.detach(),
                   f"{split}/kl_loss": kl_loss.detach().mean(),
                   f"{split}/nll_loss": nll_loss.detach().mean(),
                   f"{split}/rec_loss": rec_loss.detach().mean(),
                   f"{split}/d_weight": d_weight.detach(),
                   f"{split}/disc_factor": torch.tensor(disc_factor),
                   f"{split}/g_loss": g_loss.detach().mean(),
                   f"{split}/writer_loss": writer_loss.detach().mean(),
                   f"{split}/HTR_loss": htr_loss.detach().mean(),
                   f"{split}/HTR_rec_loss": htr_rec_loss.detach().mean(),
                   f"{split}/cer": cer.detach().mean(),
                   f"{split}/acc": acc.detach().mean(),
                   }

            return loss, log

        elif optimizer_idx == 1:
            # GAN discriminator update
            logits_real = self.discriminator(images.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                f"{split}/disc_loss": d_loss.clone().detach().mean(),
                f"{split}/logits_real": logits_real.detach().mean(),
                f"{split}/logits_fake": logits_fake.detach().mean()
            }
            return d_loss, log



    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight
