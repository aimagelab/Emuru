from torch.utils.data import Dataset
from . import font_transforms as FT
from torchvision import transforms as T
from pathlib import Path
import nltk
from einops import rearrange
from torch.nn.utils.rnn import pad_sequence


def pad_images(images, padding_value=1):
    images = [rearrange(img, 'c h w -> w c h') for img in images]
    return rearrange(pad_sequence(images, padding_value=padding_value), 'w b c h -> b c h w')


class OnlineFontSquare(Dataset):
    def __init__(self, fonts_path, backgrounds_path, text_sampler=None, transform=None):
        self.fonts_path = Path(fonts_path)
        self.fonts = list(Path(fonts_path).glob('*.ttf'))
        self.text_sampler = text_sampler
        self.transform = T.Compose([
            FT.RenderImage(self.fonts, calib_threshold=0.8, pad=20),
            T.RandomRotation(3, fill=1),
            FT.RandomWarping(grid_shape=(5, 2)),
            T.GaussianBlur(kernel_size=3),
            FT.RandomBackground(Path(backgrounds_path)),
            FT.TailorTensor(pad=3),
            FT.ToCustomTensor(),
            # FT.GrayscaleErosion(kernel_size=2, p=0.05),
            FT.GrayscaleDilation(kernel_size=2, p=0.1),

            FT.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0),
            FT.ImgResize(64),
            FT.Normalize((0.5,), (0.5,))
        ]) if transform is None else transform

        self.lenght = 1000

    def __len__(self):
        return self.lenght
    
    def __getitem__(self, _):
        text = self.text_sampler()
        img, bw_img = self.transform(text)
        return img, bw_img, text
    
    def collate_fn(self, batch):
        imgs, bw_imgs, texts = zip(*batch)
        imgs = pad_images(imgs)
        bw_imgs = pad_images(bw_imgs)
        return imgs, bw_imgs, texts
    

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
            res = self.words[self.idx:self.idx+self.count]
            self.idx += self.count
        
        txt = ' '.join(res)
        if self.min_len is not None and len(txt) < self.min_len:
            txt = txt + (' ' * (self.min_len - len(txt)))
        if self.max_len is not None and len(txt) > self.max_len:
            txt = txt[:self.max_len]
        return txt




