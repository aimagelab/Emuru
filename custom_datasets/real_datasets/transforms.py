from PIL import Image
import random
import math
from torchvision.transforms import functional as F
from torchvision import transforms as T
from torchvision.transforms import Compose
import torch


class ResizeFixedHeight(object):
    def __init__(self, height):
        self.height = height

    def __call__(self, sample):
        img, lbl = sample
        w, h = img.size
        ratio = h / self.height
        new_w = int(w / ratio)
        img = img.resize((new_w, self.height), Image.BILINEAR)
        return img, lbl


class RandomShrink(object):
    def __init__(self, min_ratio, max_ratio, min_width=0, max_width=10 ** 9, snap_to=1):
        assert min_ratio <= max_ratio
        assert min_width <= max_width
        assert snap_to > 0
        assert min_width % snap_to == 0
        assert max_width % snap_to == 0
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.min_width = min_width
        self.max_width = max_width
        self.sanp_to = snap_to

    def __call__(self, sample):
        img, lbl = sample
        w, h = img.size
        min_width = min(max(int(w * self.min_ratio), self.min_width), self.max_width)
        max_width = max(min(int(w * self.max_ratio), self.max_width), self.min_width)
        assert min_width <= max_width, f'{min_width=} > {max_width=}'
        new_w = random.randint(min_width, max_width)
        new_w = math.ceil(new_w / self.sanp_to) * self.sanp_to
        img = img.resize((new_w, h), Image.BILINEAR)
        assert img.size[0] >= self.min_width, f'{img.size[0]} < {self.min_width}'
        assert img.size[0] <= self.max_width, f'{img.size[0]} > {self.max_width}'
        return img, lbl
    

class MedianRemove(object):
    def __call__(self, sample):
        img, lbl = sample
        c, *_ = img.shape
        median_values = img.view(c, -1).median(-1)[0]
        img = img + (1 - median_values) * random.random()
        img = img.clamp(0, 1)
        return img, lbl
    

class RandomCrop(T.RandomCrop):
    def __call__(self, sample):
        img, lbl = sample
        return super().__call__(img), lbl


class PadNextDivisible(object):
    def __init__(self, divisible, padding_value=1):
        self.divisible = divisible
        self.padding_value = padding_value

    def __call__(self, sample):
        img, lbl = sample
        width = img.shape[-1]
        if width % self.divisible == 0:
            return img, lbl
        pad_width = self.divisible - width % self.divisible
        return F.pad(img, (0, 0, pad_width, 0), fill=self.padding_value), lbl
    
    
class CropMaxWidth(object):
    def __init__(self, max_width):
        self.max_width = max_width

    def __call__(self, sample):
        img, lbl = sample
        return img[..., :self.max_width], lbl


class Convert(object):
    def __init__(self, channels):
        if channels == 1:
            self.mode = 'L'
        elif channels == 3:
            self.mode = 'RGB'
        else:
            raise NotImplementedError

    def __call__(self, sample):
        img, lbl = sample
        return img.convert(self.mode), lbl


class ToTensor(T.ToTensor):
    def __call__(self, sample):
        img, lbl = sample
        return super().__call__(img), lbl


class ToPILImage(T.ToPILImage):
    def __call__(self, sample):
        img, lbl = sample
        return super().__call__(img), lbl


class Normalize(T.Normalize):
    def __call__(self, sample):
        img, lbl = sample
        return super().__call__(img), lbl


class FixedCharWidth(object):
    def __init__(self, width):
        self.width = width

    def __call__(self, sample):
        img, lbl = sample
        w, h = img.size
        new_w = self.width * len(lbl)
        img = img.resize((new_w, h), Image.BILINEAR)
        return img, lbl


class PadMinWidth(object):
    def __init__(self, min_width, padding_value=1):
        self.min_width = min_width
        self.padding_value = padding_value

    def __call__(self, sample):
        img, lbl = sample
        if isinstance(img, Image.Image):
            w, h = img.size
        elif isinstance(img, torch.Tensor):
            c, h, w = img.shape
        else:
            raise NotImplementedError
        
        if w >= self.min_width:
            return img, lbl
        pad_width = self.min_width - w
        return F.pad(img, (0, 0, pad_width, 0), fill=self.padding_value), lbl
