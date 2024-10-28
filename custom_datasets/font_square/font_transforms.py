import pickle
import time
from typing import Sequence
from .render_font import Render
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import random
import os
from PIL import Image
from .tps import TPS
import json
import string
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
import pickle
from io import BytesIO


def mask_coords(mask):
    canvas = torch.zeros((mask.shape[1] + 2, mask.shape[2] + 2))
    canvas[1:-1, 1:-1] = mask
    x0 = canvas.max(0).values.int().argmax()
    y0 = canvas.max(1).values.int().argmax()
    x1 = canvas.shape[1] - canvas.max(0).values.flip(0).int().argmax()
    y1 = canvas.shape[0] - canvas.max(1).values.flip(0).int().argmax()
    return x0 + 1, y0 + 1, x1 - 1, y1 - 1


class RenderImage(object):
    def __init__(self, fonts_path, renderers, pad=0):
        self.pad = pad

        # fonts_data_path = fonts_path[0].parent / 'fonts_sizes.json'
        # if fonts_data_path.exists():
        #     with open(fonts_data_path, 'r') as f:
        #         fonts_data = json.load(f)
        # else:
        #     fonts_data = {}

        self.renderers = renderers
        self.fonts_to_ids = {path.name: i for i, path in enumerate(fonts_path)}
        self.ids_to_fonts = {i: path for i, path in enumerate(fonts_path)}

        # with open(fonts_data_path, 'w') as f:
        #     json.dump(fonts_data, f)

    def __call__(self, sample):
        font_id = sample['font_id'] if 'font_id' in sample else random.randrange(len(self.renderers))
        render_class = self.renderers[font_id]
        try:
            np_img, sample['text'] = render_class.render(sample['text'], action='top_left', pad=self.pad)
        except OSError:
            try:
                print(f'WARNING rendering "{sample["text"]}" with font {self.ids_to_fonts[font_id].name}. Try to render only ascii letters.')
                sample['text'] = ''.join([c for c in sample['text'] if c in set(string.ascii_lowercase + ' ')])
                sample['text'] = ' '.join(sample['text'].split())
                np_img, sample['text'] = render_class.render(sample['text'], action='top_left', pad=self.pad)
            except OSError:
                print(f'ERROR rendering "{sample["text"]}" with font {self.ids_to_fonts[font_id].name}. Rendering with empty text. Please consider removing this font.')
                print(f'\t- Font path: {self.ids_to_fonts[font_id].absolute()}')
                sample['text'] = ''
                np_img = np.zeros((128, 64), dtype=np.uint8)
        except Exception as e:
            print(f'FATAL ERROR rendering "{sample["text"]}" with font {self.ids_to_fonts[font_id].name}.')
            print(f'\t- Font path: {self.ids_to_fonts[font_id].absolute()}')
            print(f'\t- Error: {e}')
            sample['text'] = ''
            np_img = np.zeros((128, 64), dtype=np.uint8)


        sample['img'] = torch.from_numpy(np_img).unsqueeze(0).float()
        return sample


class RandomWarping:
    def __init__(self, std=0.05, grid_shape=(5, 3), p=0.5):
        self.std = std
        self.grid_shape = grid_shape
        self.p = p

    def __call__(self, sample):
        if random.random() > self.p:
            return sample

        img = sample['img']
        _, h, w = img.shape
        x = np.linspace(-1, 1, self.grid_shape[0])
        y = np.linspace(-1, 1, self.grid_shape[1])
        xx, yy = np.meshgrid(x, y)

        # make source surface, get uniformed distributed control points
        source_xy = np.stack([xx, yy], axis=2).reshape(-1, 2)

        # make deformed surface
        deform_xy = source_xy + np.random.normal(scale=self.std, size=source_xy.shape)
        # deform_xy = np.stack([xx, yy], axis=2).reshape(-1, 2)

        # get coefficient, use class
        trans = TPS(source_xy, deform_xy)

        # make other points a left-bottom to upper-right line on source surface
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(x, y)
        test_xy = np.stack([xx, yy], axis=2).reshape(-1, 2)

        # get transformed points
        transformed_xy = trans(test_xy)
        grid = torch.from_numpy(transformed_xy)
        grid = grid.reshape(1, h, w, 2)
        img = img.unsqueeze(0).type(grid.dtype)
        sample['img'] = torch.nn.functional.grid_sample(img, grid, mode='nearest', padding_mode='border',
                                                        align_corners=False).squeeze(0).float()
        return sample


class GaussianBlur(T.GaussianBlur):
    def __init__(self, kernel_size, p=0.5):
        super().__init__(kernel_size)
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample['img'] = super().forward(sample['img'])
        return sample


class ImgResize:
    def __init__(self, height):
        self.height = height

    def __call__(self, sample):
        _, h, w = sample['text_img'].shape
        if h == self.height:
            return sample
        out_w = int(self.height * w / h)
        # assert out_w > 0 and out_w < 10_000, f'Invalid width {out_w} for image size {h}x{w}. Font: {sample["font_id"]} Text: {sample["text"]}'

        sample['bg_patch'] = F.resize(sample['bg_patch'], [self.height, out_w], antialias=True)
        sample['text_img'] = F.resize(sample['text_img'], [self.height, out_w], antialias=True)
        return sample


class MaxWidth:
    def __init__(self, width):
        self.width = width

    def __call__(self, sample):
        _, h, w = sample['img'].shape
        if w <= self.width:
            return sample

        sample['img'] = sample['img'][:, :, :self.width]
        sample['alpha'] = sample['alpha'][:, :, :self.width]
        sample['bg_patch'] = sample['bg_patch'][:, :, :self.width]

        return sample

class ToWidth:
    def __init__(self, width):
        self.width = width

    def __call__(self, sample):
        _, h, w = sample['text_img'].shape
        if w == self.width:
            return sample
        elif w < self.width:
            pad_w = self.width - w
            sample['bg_patch'] = F.pad(sample['bg_patch'], (0, 0, pad_w, 0), fill=1)
            sample['text_img'] = F.pad(sample['text_img'], (0, 0, pad_w, 0), fill=0)
        else:
            sample['bg_patch'] = sample['bg_patch'][:, :, :self.width]
            sample['text_img'] = sample['text_img'][:, :, :self.width]

        return sample

class PadDivisible:
    def __init__(self, divisor):
        self.divisor = divisor

    def __call__(self, sample):
        _, h, w = sample['img'].shape
        pad_w = (self.divisor - w % self.divisor) % self.divisor
        if pad_w == 0:
            return sample

        sample['img'] = F.pad(sample['img'], (0, 0, pad_w, 0), fill=1)
        sample['alpha'] = F.pad(sample['alpha'], (0, 0, pad_w, 0), fill=1)
        sample['bg_patch'] = F.pad(sample['bg_patch'], (0, 0, pad_w, 0), fill=1)
        return sample


class RandomBackground(object):
    start_time = time.time()

    def __init__(self, backgrounds, white_p=0.5):
        self.bgs_paths = backgrounds
        self.bgs = [F.to_tensor(Image.open(path).convert('RGB')) for path in self.bgs_paths]
        self.white_p = white_p

    def get_available_idx(self, img_h, img_w):
        return [(idx, bg) for idx, bg in enumerate(self.bgs) if bg.shape[1] >= img_h and bg.shape[2] >= img_w]

    @staticmethod
    def random_patch(bg, img_h, img_w):
        _, bg_h, bg_w = bg.shape
        # assert bg_h >= img_h and bg_w >= img_w, f'Background size {bg_h}x{bg_w} is too small for image size {img_h}x{img_w}'
        resize_crop = T.RandomResizedCrop(
            (img_h, img_w),
            scale=(1.0, 1.0),
            ratio=(1.0, 1.0),
            antialias=True
            )
        i, j, h, w = resize_crop.get_params(bg, resize_crop.scale, resize_crop.ratio)
        return F.resized_crop(bg, i, j, h, w, resize_crop.size, resize_crop.interpolation, antialias=True), (i, j, h, w)

    def __call__(self, sample):
        _, h, w = sample['img'].shape
        
        if 'bg_idx' not in sample:
            available_bgs = self.get_available_idx(h, w)
            if len(available_bgs) == 0 or random.random() < self.white_p:
                sample['bg_patch'] = torch.ones((3, h, w))
                return sample
            bg_idx, bg = random.choice(available_bgs)
        else:
            bg_idx = sample['bg_idx']
            if bg_idx < 0:
                sample['bg_patch'] = torch.ones((3, h, w))
                return sample
            bg = self.bgs[bg_idx]

        sample['bg_path'] = self.bgs_paths[bg_idx]
        bg_patch, _ = self.random_patch(bg, h, w)
        assert bg_patch.numel() > 0, f'Background patch is empty for image size {h}x{w}'

        if random.random() > 0.5:
            bg_patch = bg_patch.flip(1)  # up-down
        if random.random() > 0.5:
            bg_patch = bg_patch.flip(2)  # left-right

        sample['bg_patch'] = bg_patch
        return sample


class TailorTensor:
    def __init__(self, pad=0):
        self.pad = pad

    def __call__(self, sample):
        img, bg_patch = sample['img'], sample['bg_patch']
        x0, y0, x1, y1 = mask_coords(img < 0.99)
        x0 = max(0, x0 - self.pad)
        x1 = min(bg_patch.shape[2], x1 + self.pad)
        y0 = max(0, y0 - self.pad)
        y1 = min(bg_patch.shape[1], y1 + self.pad)

        img = img[:, y0:y1, x0:x1]
        bg_patch = bg_patch[:, y0:y1, x0:x1]

        sample['img'], sample['bg_patch'] = img, bg_patch
        return sample


class SplitAlphaChannel:
    def __init__(self, min_alpha=0.5, max_alpha=1.0):
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

    def __call__(self, sample):
        bw_img = sample['img']
        alpha_ink = random.uniform(self.min_alpha, self.max_alpha)
        sample['alpha_ink'] = alpha_ink
        sample['text_img'] = 1 - bw_img

        # alpha  = (1 - bw_img) * alpha_ink 
        # rgb_img = 1 - torch.cat([bw_img, bw_img, bw_img], dim=0)
        #  sample['greyscale_img'] = 1 - bw_img
        # color = torch.rand(3)
        # sample['rgba_img'] = torch.cat([rgb_img * color[:, None, None], alpha], dim=0)
        
        return sample
    

class MergeWithBackground:
    def __call__(self, sample):
        # img = sample['text_img'] * sample['alpha_ink'] + sample['bg_patch'] * (1 - sample['alpha_ink'])
        sample['text_img'] = 1 - (sample['text_img'] * sample['alpha_ink'])
        sample['img']  = sample['text_img'] * sample['bg_patch']
        return sample



class Normalize(T.Normalize):
    def forward(self, sample):
        sample['bg_patch'] = super().forward(sample['bg_patch'])
        sample['img'] = super().forward(sample['img'])
        sample['text_img'] = super().forward(sample['text_img'])
        return sample


class RandomRotation(T.RandomRotation):
    def __init__(self, *args, p=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p

    def forward(self, sample):
        if random.random() < self.p:
            sample['img'] = super().forward(sample['img'])
        return sample


class ColorJitter(T.ColorJitter):
    def __init__(self, *args, p=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p

    def forward(self, sample):
        if random.random() < self.p:
            params = self.get_params(self.brightness, self.contrast, self.saturation, self.hue)

            sample['bg_patch'] = self.apply(sample['bg_patch'], *params)
            sample['text_img'] = self.apply(sample['text_img'], *params)

        return sample
    
    
    def apply(self, img, fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor):

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = F.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = F.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = F.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = F.adjust_hue(img, hue_factor)

        return img


class RandomGrayscale(T.RandomGrayscale):
    def forward(self, sample):
        sample['img'] = super().forward(sample['img'])
        return sample


class GrayscaleErosion:
    def __init__(self, kernel_size=5, p=0.5):
        self.kernel_size = kernel_size
        self.p = p

    def erode(self, img):
        pad = self.kernel_size // 2
        img = torch.nn.functional.pad(img, (pad, pad, pad, pad), mode='constant', value=0.0)
        img = torch.nn.functional.max_pool2d(img, self.kernel_size, stride=1)
        return img

    def __call__(self, sample):
        if random.random() > self.p:
            sample['img'] = self.erode(sample['img'])
            sample['alpha'] = self.erode(sample['alpha'])
            sample['bg_patch'] = self.erode(sample['bg_patch'])
        return sample


class GrayscaleDilation:
    def __init__(self, kernel_size=5, p=0.5):
        self.kernel_size = kernel_size
        self.p = p

    def dilate(self, img):
        pad = self.kernel_size // 2
        img = torch.nn.functional.pad(-img, (pad, pad, pad, pad), mode='constant', value=-1.0)
        img = -torch.nn.functional.max_pool2d(img, self.kernel_size, stride=1)
        return img

    def __call__(self, sample):
        if random.random() > self.p:
            sample['text_img'] = self.dilate(sample['text_img'])
            sample['bg_patch'] = self.dilate(sample['bg_patch'])
        return sample
    

class RandomInvert:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample['bg_patch'] = 1 - sample['bg_patch']
        return sample


class TimedCompose:
    def __init__(self, transforms: Sequence):
        self.transforms = transforms
        self.times = defaultdict(list)
        self.avg_times = defaultdict(list)

    def __call__(self, sample):
        for t in self.transforms:
            start = time.time()
            sample = t(sample)
            self.times[t.__class__.__name__].append(time.time() - start)
            self.avg_times[t.__class__.__name__] = np.mean(self.times[t.__class__.__name__])
        return sample

    def print_times(self):
        max_width = max(len(k) for k in self.times.keys())
        self.times = {k: np.mean(v) for k, v in self.times.items()}
        total_time = sum(self.times.values())
        for k, v in self.times.items():
            print(f'{k.ljust(max_width)} {v:.05f}s ({v / total_time:.02%})')
        print(f'Total time: {total_time:.05f}s')
        self.times = defaultdict(list)
