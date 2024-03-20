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


def mask_coords(mask):
    x0 = mask.max(1).values.type(torch.uint8).argmax()
    y0 = mask.max(2).values.type(torch.uint8).argmax()
    x1 = mask.shape[2] - mask.max(1).values.flip(0).type(torch.uint8).argmax()
    y1 = mask.shape[1] - mask.max(2).values.flip(0).type(torch.uint8).argmax()
    return x0, y0, x1, y1


class RenderImage(object):
    def __init__(self, fonts_path, height=None, width=None, calib_text=None, calib_threshold=0.7, calib_h=128, pad=0):
        self.width = width
        self.height = height
        self.pad = pad
        self.calib_text = calib_text
        self.calib_threshold = calib_threshold
        self.calib_h = calib_h

        fonts_data_path = fonts_path[0].parent / 'font_data.json'
        if fonts_data_path.exists():
            with open(fonts_data_path, 'r') as f:
                fonts_data = json.load(f)
        else:
            fonts_data = {}

        def render_fn(font_path):
            font_size = fonts_data[font_path.name] if font_path.name in fonts_data else 64
            render = Render(font_path, height, width, font_size)
            fonts_data[font_path.name] if font_path.name not in fonts_data else render.calibrate(calib_text, calib_threshold, calib_h)
            return render
        
        self.renderers = [render_fn(path) for path in fonts_path]

        with open(fonts_data_path, 'w') as f:
            json.dump(fonts_data, f)


    def __call__(self, text):
        render_class = random.choice(self.renderers)
        np_img = render_class.render(text, return_np=True, action='top_left', pad=self.pad)
        return torch.from_numpy(np_img).unsqueeze(0).float()


class RandomResizedCrop:
    def __init__(self, ratio_eps, scale):
        self.scale = scale
        self.ratio_eps = ratio_eps

    def __call__(self, sample):
        img = sample['font_img']
        ratio = img.shape[1] / img.shape[0]
        ratio = (ratio - self.ratio_eps, ratio + self.ratio_eps)

        i, j, h, w = T.RandomResizedCrop.get_params(img, self.scale, ratio)
        sample['font_img'] = F.resized_crop(img.unsqueeze(0), i, j, h, w, img.shape).squeeze(0)
        sample.record(self, ijhw=(i, j, h, w))
        return sample


class RandomWarping:
    def __init__(self, std=0.05, grid_shape=(5,3)):
        self.std = std
        self.grid_shape = grid_shape

    def __call__(self, img):
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
        img = torch.nn.functional.grid_sample(img, grid, mode='nearest', padding_mode='border', align_corners=False).squeeze(0)
        return img

class ToRGB:
    def __call__(self, img):
        rgb_img = Image.new("RGB", img.size)
        rgb_img.paste(img)
        return rgb_img

class FixedWidth:
    def __init__(self, width):
        self.width = width

    def __call__(self, img):
        c, h, w = img.shape
        canvas = torch.zeros((c, h, self.width))
        width = min(w, self.width)
        canvas[:, :, :width] = img[:, :, :width]
        return canvas


class FixedHeight:
    def __init__(self, height):
        self.height = height

    def __call__(self, img):
        c, h, w = img.shape
        canvas = torch.zeros((c, self.height, w))
        height = min(h, self.height)
        canvas[:, :height, :] = img[:, :height, :]
        return canvas


class ImgResize:
    def __init__(self, height):
        self.height = height

    def __call__(self, sample):
        img, bw_img = sample
        _, h, w = img.shape
        if h == self.height:
            return img, bw_img
        out_w = int(self.height * w / h)

        img = F.resize(img, [self.height, out_w])
        bw_img = F.resize(bw_img, [self.height, out_w])
        return img, bw_img


class RandomBackground(object):
    start_time = time.time()
    def __init__(self, backgrounds_path, include_white=True):
        self.bgs = [Image.open(path) for path in backgrounds_path.rglob('*.png')]
        self.bgs = [F.to_tensor(img.convert('RGB')) for img in self.bgs]
        self.include_white = include_white
        if include_white:
            max_height = max([bg.shape[1] for bg in self.bgs])
            max_width = max([bg.shape[2] for bg in self.bgs])
            self.bgs.append(torch.ones((3, max_height, max_width), dtype=torch.float32))

    @staticmethod
    def random_patch(bg, img_h, img_w):
        _, bg_h, bg_w = bg.shape
        resize_crop = T.RandomResizedCrop((img_h, img_w), antialias=True)
        i, j, h, w = resize_crop.get_params(bg, resize_crop.scale, resize_crop.ratio)
        return F.resized_crop(bg, i, j, h, w, resize_crop.size, resize_crop.interpolation), (i, j, h, w)


    def __call__(self, img):
        _, h, w = img.shape

        bg_idx = random.randrange(0, len(self.bgs))
        bg_patch, _ = self.random_patch(self.bgs[bg_idx], h, w)

        if random.random() > 0.5:
            bg_patch = bg_patch.flip(1)   # up-down
        if random.random() > 0.5:
            bg_patch = bg_patch.flip(2)   # left-right

        return img, bg_patch

class TailorTensor:
    def __init__(self, pad=0):
        self.pad = pad

    def __call__(self, sample):
        img, bg_patch = sample
        x0, y0, x1, y1 = mask_coords(img < 0.99)
        x0 = max(0, x0 - self.pad)
        x1 = min(bg_patch.shape[2], x1 + self.pad)
        y0 = max(0, y0 - self.pad)
        y1 = min(bg_patch.shape[1], y1 + self.pad)

        img = img[:, y0:y1, x0:x1]
        bg_patch = bg_patch[:, y0:y1, x0:x1]
        return img, bg_patch

class ToCustomTensor:
    def __init__(self, min_alpha=0.5, max_alpha=1.0):
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

    def __call__(self, sample):
        bw_img, bg_patch = sample
        alpha = random.uniform(self.min_alpha, self.max_alpha)
        font_mask = 1 - ((1 - bw_img) * alpha)
        img = font_mask * bg_patch
        return img, bw_img


class Normalize(T.Normalize):
    def forward(self, sample):
        img, bw_img = sample
        img = super().forward(img)
        img_bw = super().forward(bw_img)
        return img, img_bw


class RandomRotation(T.RandomRotation):
    def __init__(self, degrees, *args, **kwargs):
        super().__init__(degrees, *args, **kwargs)

    def forward(self, img):
        angle = self.get_params(self.degrees)
        img = F.rotate(1 - img, angle, F.InterpolationMode.NEAREST, self.expand, self.center, self.fill)
        return 1 - img


class ColorJitter(T.ColorJitter):
    def forward(self, sample):
        img, bw_img = sample
        img = super().forward(img)
        return img, bw_img


# class RandomAdjustSharpness(T.RandomAdjustSharpness):
#     def forward(self, sample):
#         text, img = sample
#         return text, super().forward(img)


class RandomGrayscale(T.RandomGrayscale):
    def forward(self, sample):
        sample['img'] = super().forward(sample['img'])
        return sample


# class RandomSolarize(T.RandomSolarize):
#     def forward(self, sample):
#         text, img = sample
#         return text, super().forward(img)


# class RandomInvert(T.RandomSolarize):
#     def __init__(self, p=0.5):
#         super().__init__(0, p)
#
#     def forward(self, sample):
#         text, img = sample
#         return text, super().forward(img)


class RandomAffine(T.RandomAffine):
    def forward(self, sample):
        text, img = sample
        channels = img.shape[0]
        if channels == 3:
            img = super().forward(img)
        elif channels == 4:
            img[0] = super().forward(img[0].unsqueeze(0))
        else:
            raise NotImplementedError
        return text, img

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
        img, bw_img = sample
        if random.random() > self.p:
            img = self.erode(img)
            bw_img = self.erode(bw_img)
        return img, bw_img

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
        img, bw_img = sample
        if random.random() > self.p:
            img = self.dilate(img)
            bw_img = self.dilate(bw_img)
        return img, bw_img

class SaveHistory:
    def __init__(self, out_dir, out_type):
        self.out_dir = out_dir
        self.out_type = out_type
        os.makedirs(out_dir, exist_ok=True)

    def __call__(self, sample):
        path = os.path.join(self.out_dir, sample['text'])
        if self.out_type == 'json':
            with open(path + '.json', 'w') as f:
                json.dump(sample.to_dict(), f)
        elif self.out_type == 'pickle':
            with open(os.path.join(self.out_dir, sample['text']) + '.pkl', 'wb') as f:
                pickle.dump(sample.to_dict(), f)
        elif self.out_type == 'png':
            F.to_pil_image(sample['img']).save(path + '.png')
        elif self.out_type == 'jpg':
            F.to_pil_image(sample['img']).save(path + '.jpg')
        else:
            raise NotImplementedError
        return sample