from torch.utils.data import Dataset
from . import font_transforms as FT
from torchvision import transforms as T
from pathlib import Path
import nltk
import torch
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence


def pad_images(images, padding_value=1):
    images = [rearrange(img, 'c h w -> w c h') for img in images]
    return rearrange(pad_sequence(images, padding_value=padding_value), 'w b c h -> b c h w')


class OnlineFontSquare(Dataset):
    def __init__(self, fonts_path, backgrounds_path, text_sampler=None, transform=None, length=None):
        self.fonts_path = Path(fonts_path)
        self.fonts = list(Path(fonts_path).glob('*.ttf'))
        self.text_sampler = text_sampler
        self.transform = T.Compose([
            FT.RenderImage(self.fonts, calib_threshold=0.8, pad=20),
            FT.RandomRotation(3, fill=1),
            FT.RandomWarping(grid_shape=(5, 2)),
            FT.GaussianBlur(kernel_size=3),
            FT.RandomBackground(Path(backgrounds_path)),
            FT.TailorTensor(pad=3),
            FT.ToCustomTensor(),
            # FT.GrayscaleErosion(kernel_size=2, p=0.05),
            FT.GrayscaleDilation(kernel_size=2, p=0.1),

            FT.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0),
            FT.ImgResize(64),
            FT.MaxWidth(768),
            FT.Normalize((0.5,), (0.5,))
        ]) if transform is None else transform

        self.length = len(self.fonts) if length is None else length

    def __len__(self):
        return self.length

    def __getitem__(self, font_id):
        text = self.text_sampler()
        sample = self.transform({'text': text, 'font_id': font_id})
        return sample

    def collate_fn(self, batch):
        collate_batch = {}

        for key in batch[0].keys():
            val = batch[0][key]
            if isinstance(val, torch.Tensor):
                collate_batch[key] = pad_images([sample[key] for sample in batch])
            elif isinstance(val, int):
                collate_batch[key] = torch.IntTensor([sample[key] for sample in batch])
            elif isinstance(val, float):
                collate_batch[key] = torch.FloatTensor([sample[key] for sample in batch])
            elif isinstance(val, bool):
                collate_batch[key] = torch.BoolTensor([sample[key] for sample in batch])
            else:
                collate_batch[key] = [sample[key] for sample in batch]

        return collate_batch


class TextSampler:
    def __init__(self, min_len, max_len, count, exponent=1, charset=None):
        self.min_len = min_len
        self.max_len = max_len
        self.charset = charset
        self.exponent = exponent
        self.count = count

        self.idx = 0
        self.load_words()

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
        if self.idx + self.count > len(self.words):
            res = self.words[self.idx:]
            res += self.words[:self.count - len(res)]
            self.idx = self.count - len(res)
        else:
            res = self.words[self.idx:self.idx + self.count]
            self.idx += self.count

        txt = ' '.join(res)
        if self.min_len is not None and len(txt) < self.min_len:
            txt = txt + (' ' * (self.min_len - len(txt)))
        if self.max_len is not None and len(txt) > self.max_len:
            txt = txt[:self.max_len]
        return txt
