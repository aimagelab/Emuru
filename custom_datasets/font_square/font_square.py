from torch.utils.data import Dataset
from . import font_transforms as FT
from torchvision import transforms as T
from pathlib import Path
import numpy as np
import nltk
import torch
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence
import random
import msgpack
from pathlib import Path
from collections import Counter
from itertools import pairwise
import tarfile

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

    images = pad_images(images, padding_value=PAD)
    images_bw = pad_images(images_bw, padding_value=PAD)
    text_logits_ctc = pad_sequence(text_logits_ctc, padding_value=PAD, batch_first=True)
    text_logits_s2s = pad_sequence(text_logits_s2s, padding_value=PAD, batch_first=True)
    tgt_key_mask = subsequent_mask(text_logits_s2s.shape[-1] - 1)
    tgt_key_padding_mask = text_logits_s2s == PAD

    return {
        'images': images,
        'images_bw': images_bw,
        'texts': texts,
        'unpadded_texts_len': torch.LongTensor(texts_len),
        'writers': torch.LongTensor(writers),
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
        extract_path = 'files/font_square/extracted_fonts'

        if isinstance(fonts, Path) and fonts.suffix == '.gz':
            with tarfile.open(fonts, 'r:gz') as tar:
                tar.extractall(path=extract_path)
            fonts = Path(extract_path)

        if isinstance(fonts, Path) and fonts.is_dir():
            self.fonts = sorted(list(fonts.glob('*.?tf')))
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
            FT.RenderImage(self.fonts, calib_threshold=0.8, pad=20, verbose=True),
            FT.RandomRotation(3, fill=1, p=0.5),
            FT.RandomWarping(grid_shape=(5, 2), p=0.15),
            FT.GaussianBlur(kernel_size=3, p=0.5),
            FT.RandomBackground(backgrounds, white_p=0.5),
            FT.TailorTensor(pad=3),
            FT.MergeWithBackground(),
            # FT.GrayscaleErosion(kernel_size=2, p=0.05),
            FT.GrayscaleDilation(kernel_size=2, p=0.1),
            FT.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0, p=0.5),
            FT.RandomInvert(p=0.3),
            FT.ImgResize(64),
            # FT.MaxWidth(768),
            FT.ToWidth(768),
            # FT.PadDivisible(8),
            FT.Normalize((0.5,), (0.5,))
        ]) if transform is None else transform

        self.length = len(self.fonts) if length is None else length
        self.alphabet = Alphabet(charset=FONT_SQUARE_CHARSET)

    def __len__(self):
        return self.length

    def __getitem__(self, font_id):
        font_id = font_id % len(self.fonts)
        text = self.text_sampler()
        sample = self.transform({'text': text, 'font_id': font_id})
        sos = torch.LongTensor([START_OF_SEQUENCE])
        eos = torch.LongTensor([END_OF_SEQUENCE])
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
        res['img'] = pad_images([sample['image'] for sample in batch])
        return res


class TextSampler:
    def __init__(self, min_len: int, max_len: int, count, charset=None, exponent=1,
                 words_dict_path='files/font_square/words_dict.msgpack'):

        self.min_len = min_len
        self.max_len = max_len
        self.charset = charset
        self.exponent = exponent

        if isinstance(count, int):
            self.min_count = count
            self.max_count = count
        elif isinstance(count, (tuple, list)):
            self.min_count = count[0]
            self.max_count = count[1]

        self.load_words(words_dict_path)
        self.num_words = len(self.words)

    def load_words(self, words_dict_path):
        if Path(words_dict_path).exists():
            with open(words_dict_path, 'rb') as file:
                packed_data = file.read()
                words_frequencies_dict = msgpack.unpackb(packed_data)
            self.words = list(words_frequencies_dict.keys())
            assert len(self.words) == 103411, 'Words count mismatch. Expected 103411 words.'
            # self.words_frequencies = torch.tensor(list(words_frequencies_dict.values()), dtype=torch.float)
        else:
            words = nltk.corpus.abc.words()
            words += nltk.corpus.brown.words()
            words += nltk.corpus.genesis.words()
            words += nltk.corpus.inaugural.words()
            words += nltk.corpus.state_union.words()
            words += nltk.corpus.webtext.words()

            if self.charset is not None:
                words = [word for word in words if all([c in self.charset for c in word])]

            words = list(words)
            words_unique = list(set(words))
            words_count = Counter(words_unique)
            words_frequencies = {word: count for word, count in words_count.items()}
            with open(words_dict_path, 'wb') as file:
                packed_data = msgpack.packb(words_frequencies)
                file.write(packed_data)

            self.words = words_unique
            # self.words_frequencies = torch.tensor(list(words_frequencies.values()), dtype=torch.float)

        unigram_long_text = ''.join(self.words)
        unigram_counts = Counter(unigram_long_text)
        self.unigram_counts = {k: len(unigram_long_text) / v ** self.exponent for k, v in unigram_counts.items()}

        bigram_long_text = ' ' + ' '.join(self.words) + ' '
        bigram_long_text = [''.join(pair) for pair in pairwise(bigram_long_text)]
        bigram_counts = Counter(bigram_long_text)
        self.bigram_counts = {k: len(bigram_long_text) / v ** self.exponent for k, v in bigram_counts.items()}
        self.words_weights = torch.tensor([self.eval_word(word) for word in self.words], dtype=torch.float)

    def eval_word(self, word):
        bigrams = list(pairwise(f' {word} '))
        unigram_score = sum([self.unigram_counts[c] for c in word]) / len(word)
        bigram_score = sum([self.bigram_counts[''.join(b)] for b in bigrams]) / len(bigrams)
        return (unigram_score + bigram_score) / 2


    def __call__(self):
        words_count = random.randint(self.min_count, self.max_count)
        words_indexes = torch.multinomial(self.words_weights, words_count, replacement=True)
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
