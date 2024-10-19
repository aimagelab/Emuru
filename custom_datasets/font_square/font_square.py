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
import json
from tqdm import tqdm
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..alphabet import Alphabet
from ..constants import (
    START_OF_SEQUENCE,
    END_OF_SEQUENCE,
    PAD,
    FONT_SQUARE_CHARSET
)
from ..subsequent_mask import subsequent_mask
from .render_font import Render


def pad_images(images, padding_value=1):
    images = [rearrange(img, 'c h w -> w c h') for img in images]
    return rearrange(pad_sequence(images, padding_value=padding_value), 'w b c h -> b c h w')


def collate_fn(batch):
    images = []
    text_images = []
    texts = []
    names = []
    text_logits_ctc = []
    text_logits_s2s = []
    texts_len = []
    writers = []

    for item in batch:
        images.append(item['img'])
        text_images.append(item['text_img'])
        texts.append(item['text'])
        names.append(item["name"])
        text_logits_ctc.append(item['text_logits_ctc'])
        text_logits_s2s.append(item['text_logits_s2s'])
        texts_len.append(len(item['text']))
        writers.append(item['writer'])

    images = pad_images(images, padding_value=PAD)
    text_images = pad_images(text_images, padding_value=PAD)
    text_logits_ctc = pad_sequence(text_logits_ctc, padding_value=PAD, batch_first=True)
    text_logits_s2s = pad_sequence(text_logits_s2s, padding_value=PAD, batch_first=True)
    tgt_key_mask = subsequent_mask(text_logits_s2s.shape[-1] - 1)
    tgt_key_padding_mask = text_logits_s2s == PAD

    return {
        'images': images,
        'text_images': text_images,
        'texts': texts,
        'unpadded_texts_len': torch.LongTensor(texts_len),
        'writers': torch.LongTensor(writers),
        'names': names,
        'text_logits_ctc': text_logits_ctc,
        'text_logits_s2s': text_logits_s2s,
        'tgt_key_mask': tgt_key_mask,
        'tgt_key_padding_mask': tgt_key_padding_mask,
    }


def get_fonts(fonts_path):
    fonts = Path(fonts_path) if isinstance(fonts_path, str) else fonts_path
    if isinstance(fonts, Path) and fonts.is_dir():
        fonts = sorted(list(fonts.glob('*.?tf')))
    elif isinstance(fonts, Path) and fonts.is_file():
        fonts = [fonts]
    elif isinstance(fonts, list):
        fonts = fonts
    else:
        raise ValueError(f'Fonts must be a directory or a list of paths. Got {type(fonts)}')
    return fonts


def make_renderers(fonts, height=None, width=None, calib_text=None, calib_threshold=0.7, calib_h=128, verbose=False, load_font_into_mem=False, 
                   num_threads=8):
    fonts = get_fonts(fonts)
    fonts_data_path = fonts[0].parent / 'fonts_sizes.json'
    if fonts_data_path.exists():
        with open(fonts_data_path, 'r') as f:
            fonts_data = json.load(f)
    else:
        fonts_data = {}

    fonts_charset_path = fonts[0].parent / 'fonts_charsets.json'
    with open(fonts_charset_path, 'r') as f:
        fonts_charset = json.load(f)

    def render_fn(font_path, load_font_into_mem):
        font_size = fonts_data.get(font_path.name, 64)
        charset = set(fonts_charset.get(font_path.name, []))
        render = Render(font_path, height, width, font_size, charset, load_font_into_mem)
        if font_path.name not in fonts_data:
            render.calibrate(calib_text, calib_threshold, calib_h)
        return render
    

    # def render_fn_batched(fonts_path, load_font_into_mem):
    #     renderers = []
    #     for path in fonts_path:
    #         renderers.append(render_fn(path, load_font_into_mem))
    #     return renderers
    

    # def load_fonts_parallel(fonts_path, load_font_into_mem, num_threads, verbose=True):
    #     renderers = []
    #     font_batches = list(chunk_list(fonts_path, num_threads))
    #     with ThreadPoolExecutor() as executor:
    #         futures = [executor.submit(render_fn_batched, batch, load_font_into_mem) for batch in font_batches]
            
    #         for future in tqdm(as_completed(futures), total=len(futures), desc='Loading fonts', disable=not verbose):
    #             renderers.extend(future.result())
        
    #     return renderers
    
    # def chunk_list(lst, num_threads):
    #     batch_size = math.ceil(len(lst) / num_threads)
    #     for i in range(0, len(lst), batch_size):
    #         yield lst[i:i + batch_size]

    # renderers = load_fonts_parallel(fonts, load_font_into_mem, num_threads, verbose)  # The rendered images are garbage, idk why
    renderers = [render_fn(path, load_font_into_mem) for path in tqdm(fonts, desc='Loading fonts', disable=not verbose)]  # no threads
    
    return renderers



class OnlineFontSquare(Dataset):
    def __init__(self, fonts, backgrounds, text_sampler=None, transform=None, length=None, load_font_into_mem=False, renderers=None):
        if backgrounds is None:
            backgrounds = []
        if isinstance(backgrounds, str):
            backgrounds = Path(backgrounds)
        if isinstance(backgrounds, Path) and backgrounds.is_dir():
            backgrounds = [p for p in backgrounds.rglob('*') if p.suffix in ('.jpg', '.png', '.jpeg')]
        assert isinstance(backgrounds, list), 'Backgrounds must be a directory or a list of paths'
        
        self.fonts = get_fonts(fonts)
        if renderers is None:
            renderers = make_renderers(fonts, calib_threshold=0.8, verbose=True, load_font_into_mem=load_font_into_mem)
        renderers = sorted(renderers, key=lambda r: len(r.charset), reverse=True)
        renderers = renderers[:100000]

        self.text_sampler = text_sampler
        self.transform = T.Compose([
            FT.RenderImage(self.fonts, pad=20, renderers=renderers),
            FT.RandomRotation(3, fill=1, p=0.5),
            FT.RandomWarping(grid_shape=(5, 2), p=1.0),
            FT.GaussianBlur(kernel_size=3, p=0.5),
            FT.RandomBackground(backgrounds, white_p=0.1),
            FT.TailorTensor(pad=3),
            FT.SplitAlphaChannel(),
            # FT.GrayscaleErosion(kernel_size=2, p=0.05),
            FT.GrayscaleDilation(kernel_size=2, p=0.1),
            FT.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0, p=0.5),
            FT.RandomInvert(p=0.2),
            FT.ImgResize(64),
            # FT.MaxWidth(768),
            FT.ToWidth(768),
            # FT.PadDivisible(8),
            FT.MergeWithBackground(),
            FT.Normalize((0.5,), (0.5,))
        ]) if transform is None else transform

        self.length = len(renderers) if length is None else length
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
            'text_img': sample['text_img'],
            'img': sample['img'],
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


class GibberishSampler:
    def __init__(self, lenght, charset=None):
        self.charset = [
            ' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', 
            '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 
            'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 
            'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 
            'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', 
            '\x80', '\x91', '\x92', '\x93', '\x94', '\x97', '\x99', '¡', '¢', '¦', '§', '¨', '®', '°', '´', 
            'µ', 'º', '»', 'À', 'Á', 'Â', 'Ä', 'Å', 'É', 'Ó', 'Ö', 'Ü', 'ß', 'à', 'á', 'â', 'ã', 'ä', 'å', 
            'ç', 'è', 'é', 'ê', 'ë', 'í', 'î', 'ï', 'ñ', 'ó', 'ô', 'õ', 'ö', 'ù', 'ú', 'û', 'ü', 'Ă', 'Ą', 
            'č', 'ď', 'ĺ', 'Ł', 'ŕ', 'Ś', 'Ť', 'ť', 'ż', 'ƒ', '˘', '˝', '—', '“', '”', '╜'
        ] if charset is None else charset
        self.chars_weights = torch.ones(len(self.charset), dtype=torch.float)
        self.chars_weights[0] = self.chars_weights.sum() * 0.16  # The sapce has the highest probability
        self.length = lenght


    def __call__(self):
        char_indexes = torch.multinomial(self.chars_weights, self.length, replacement=True)
        res = [self.charset[i] for i in char_indexes]
        txt = ''.join(res)
        return txt


class FixedTextSampler:
    def __init__(self, text):
        self.text = text

    def __call__(self):
        return self.text
