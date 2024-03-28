from torch.utils.data import Dataset
from . import font_transforms as FT
from torchvision import transforms as T
from pathlib import Path
import nltk
import torch
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence
import random

from ..alphabet import Alphabet
from ..constants import (
    START_OF_SEQUENCE,
    END_OF_SEQUENCE,
    PAD,
    FONT_SQUARE_CHARSET
)
from ..subsequent_mask import subsequent_mask


def pad_images(images, padding_value=1):
    images = [rearrange(img, 'c h w -> w c h') for img in images]
    return rearrange(pad_sequence(images, padding_value=padding_value), 'w b c h -> b c h w')


def collate_fn(batch):
    images = []
    images_bw = []
    texts = []
    names = []
    text_logits_ctc = []
    text_logits_s2s = []
    texts_len = []
    writers = []

    for item in batch:
        images.append(item['image'])
        images_bw.append(item['image_bw'])
        texts.append(item['text'])
        names.append(item["name"])
        text_logits_ctc.append(item['text_logits_ctc'])
        text_logits_s2s.append(item['text_logits_s2s'])
        texts_len.append(len(item['text']))
        writers.append(item['writer'])

    padding_value = 0  # TODO CHANGE IT AFTER REFACTORING
    images = pad_images(images, padding_value=padding_value)
    images_bw = pad_images(images_bw, padding_value=padding_value)
    text_logits_ctc = pad_sequence(text_logits_ctc, padding_value=padding_value, batch_first=True)
    text_logits_s2s = pad_sequence(text_logits_s2s, padding_value=padding_value, batch_first=True)
    tgt_key_mask = subsequent_mask(text_logits_s2s.shape[-1] - 1)
    tgt_key_padding_mask = text_logits_s2s == padding_value

    return {
        'images': images,
        'images_bw': images_bw,
        'texts': texts,
        'unpadded_texts_len': torch.LongTensor(texts_len),
        'names': names,
        'text_logits_ctc': text_logits_ctc,
        'text_logits_s2s': text_logits_s2s,
        'tgt_key_mask': tgt_key_mask,
        'tgt_key_padding_mask': tgt_key_padding_mask,
    }


class OnlineFontSquare(Dataset):
    def __init__(self, fonts, backgrounds, text_sampler=None, transform=None, length=None):
        fonts = Path(fonts) if isinstance(fonts, str) else fonts
        backgrounds = Path(backgrounds) if isinstance(backgrounds, str) else backgrounds

        if isinstance(fonts, Path) and fonts.is_dir():
            self.fonts = list(fonts.glob('*.ttf'))
        elif isinstance(fonts, Path) and fonts.is_file():
            self.fonts = [fonts]
        elif isinstance(fonts, list):
            self.fonts = fonts
        else:
            raise ValueError(f'Fonts must be a directory or a list of paths. Got {type(fonts)}')

        if isinstance(backgrounds, Path) and backgrounds.is_dir():
            backgrounds = [p for p in backgrounds.rglob('*') if p.suffix in ('.jpg', '.png', '.jpeg')]
        elif isinstance(backgrounds, Path) and backgrounds.is_file():
            backgrounds = [backgrounds]
        elif isinstance(backgrounds, list):
            backgrounds = backgrounds
        else:
            raise ValueError(f'Backgrounds must be a directory or a list of paths. Got {type(backgrounds)}')

        self.text_sampler = text_sampler
        self.transform = T.Compose([
            FT.RenderImage(self.fonts, calib_threshold=0.8, pad=20),
            FT.RandomRotation(3, fill=1),
            FT.RandomWarping(grid_shape=(5, 2), p=0.25),
            FT.GaussianBlur(kernel_size=3),
            FT.RandomBackground(backgrounds),
            FT.TailorTensor(pad=3),
            FT.MergeWithBackground(),
            # FT.GrayscaleErosion(kernel_size=2, p=0.05),
            FT.GrayscaleDilation(kernel_size=2, p=0.1),

            FT.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0),
            FT.ImgResize(64),
            # FT.MaxWidth(768),
            FT.ToWidth(768),
            FT.PadDivisible(8),
            FT.Normalize((0.5,), (0.5,))
        ]) if transform is None else transform

        self.length = len(self.fonts) if length is None else length
        self.alphabet = Alphabet(charset=FONT_SQUARE_CHARSET)

    def __len__(self):
        return self.length

    def __getitem__(self, font_id):
        text = self.text_sampler()
        sample = self.transform({'text': text, 'font_id': font_id})
        sos = self.alphabet.encode([START_OF_SEQUENCE])
        eos = self.alphabet.encode([END_OF_SEQUENCE])
        text_logits_ctc = self.alphabet.encode(text)
        text_logits_s2s = torch.cat([sos, text_logits_ctc, eos])
        unpadded_text_len = len(sample['text'])

        return {
            'image': sample['img'],
            'image_bw': sample['bw_img'],
            'text': text,
            'writer': font_id,
            'text_logits_ctc': text_logits_ctc,
            'text_logits_s2s': text_logits_s2s,
            'unpadded_text_len': unpadded_text_len,
            'name': f'{font_id}',
        }


class HFDataCollector:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        txts = [sample['text'] for sample in batch]
        res = self.tokenizer(txts, padding=True, return_tensors='pt', return_attention_mask=True, return_length=True)
        res['img'] = pad_images([sample['img'] for sample in batch])
        return res


class TextSampler:
    def __init__(self, min_len: int, max_len: int, count, charset=None):
        self.min_len = min_len
        self.max_len = max_len
        self.charset = charset

        if isinstance(count, int):
            self.min_count = count
            self.max_count = count
        elif isinstance(count, (tuple, list)):
            self.min_count = count[0]
            self.max_count = count[1]

        self.load_words()
        self.num_words = len(self.words)

    def load_words(self):
        self.words = nltk.corpus.abc.words()
        self.words += nltk.corpus.brown.words()
        self.words += nltk.corpus.genesis.words()
        self.words += nltk.corpus.inaugural.words()
        self.words += nltk.corpus.state_union.words()
        self.words += nltk.corpus.webtext.words()

        if self.charset is not None:
            self.words = [word for word in self.words if all([c in self.charset for c in word])]

    def __call__(self):
        words_count = random.randint(self.min_count, self.max_count)
        words_indexes = torch.randint(0, self.num_words, (words_count,))  # TODO ALSO THE NUMBER OF WORDS SHOULD CHANGE
        res = [self.words[i] for i in words_indexes]
        txt = ' '.join(res)

        if self.min_len is not None and len(txt) < self.min_len:
            txt = txt + (' ' * (self.min_len - len(txt)))
        if self.max_len is not None and len(txt) > self.max_len:
            txt = txt[:self.max_len]
        return txt


class FixedTextSampler:
    def __init__(self, text):
        self.text = text

    def __call__(self):
        return self.text
