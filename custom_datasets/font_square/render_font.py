from asyncio import constants
import os
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import time
from nltk.corpus import words
from random import sample, randint
import string
import cv2
import json
from pathlib import Path

try:
    terminal_columns = os.get_terminal_size().columns
except OSError:
    terminal_columns = 60

class Render:
    ink_path = None
    pag_path = None

    def __init__(self, font_path, height=None, width=None, font_size=64, charset=None):
        self.font_path = font_path
        self.font_size = font_size
        self.height = height
        self.width = width

        self.font_path = str(Path(font_path).absolute())

        try:
            self.font = ImageFont.truetype(self.font_path, self.font_size)
        except OSError:
            raise OSError(f'Error: {font_path}')
        
        self.charset = charset
    
    def calibrate(self, text=None, threshold=0.7, height=128, width=2500):
        text = string.ascii_letters if text is None else text
        # Reduce font size until the text fits the threshold
        while self._perc(text, height, width) > threshold:
            self.font_size -= 1
            if self.font_size <= 1: break
            self.font = ImageFont.truetype(self.font_path, self.font_size)
        # Increase font size until the text fits the threshold
        while self._perc(text, height, width) < threshold:
            self.font_size += 1
            self.font = ImageFont.truetype(self.font_path, self.font_size)
        return self.font_size
    
    def set_templates(self, pag_path, ink_path):
        self.ink_path = ink_path
        self.pag_path = pag_path
        if self.ink_path:
            self.ink_img = cv2.imread(ink_path)
            self.ink_img = cv2.cvtColor(self.ink_img, cv2.COLOR_BGR2RGB)
        if self.pag_path:
            self.pag_img = cv2.imread(pag_path)
            self.pag_img = cv2.cvtColor(self.pag_img, cv2.COLOR_BGR2RGB)

    def _get_patch(self, img, h, w):
        img_h, img_w, _ = img.shape
        assert img_h >= h and img_w >= w
        x0 = randint(0, img_h - h)
        y0 = randint(0, img_w - w)
        return img[x0:x0+h, y0:y0+w]

    def render(self, text, action='random', pad=0):
        if self.charset is not None:
            text = ''.join([c for c in text if c in self.charset])

        bbox_width, bbox_height = self._text_wh(text)
        if self.height is None or self.width is None:
            h, w =  bbox_height + 2 * pad, bbox_width + 2 * pad
        else:
            h, w = self.height, self.width
        img = np.ones((h, w), np.uint8)

        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)

        if action == 'random':
            margin_h, margin_w = h - bbox_height, w - bbox_width
            xy = (
                randint(min(0, margin_w), max(0, margin_w)),
                randint(min(0, margin_h), max(0, margin_h)),
            )
            draw.text(xy,  text, font=self.font, fill=0, anchor='lt')
        elif action == 'center':
            xy = (w // 2, h // 2)
            draw.text(xy, text, font=self.font, fill=0, anchor='mm')
        elif action == 'center_left':
            xy = (pad, h // 2)
            draw.text(xy, text, font=self.font, fill=0, anchor='lm')
        elif action == 'top_left':
            draw.text((pad, pad), text, font=self.font, fill=0, anchor='lt')
        else:
            raise NotImplementedError

        img = np.array(img_pil)
        # img = unpad(img)
        # img = np.pad(img, 30, mode='constant', constant_values=255)
        img = self._convert_colors(img)
        return img, text
    
    def __call__(self, text):
        return self.render(text)

    def _text_wh(self, text):
        bbox = self.font.getbbox(text)
        bbox_height, bbox_width = max(abs(bbox[2] - bbox[0]), 1), max(abs(bbox[3] - bbox[1]), 1)
        assert bbox_height > 0 and bbox_width > 0, f'bbox_height:{bbox_height}, bbox_width:{bbox_width}, text:{text}'
        return bbox_height, bbox_width

    def _perc(self, text, height, width):
        text_w, text_h = self._text_wh(text)
        h_perc, w_perc = text_h / height, text_w / width
        return max(h_perc, w_perc)
    
    def _convert_colors(self, img):
        if self.ink_path is not None or self.pag_path is not None:
            patch_ink = self._get_patch(self.ink_img, img.shape[0], img.shape[1])
            patch_pag = self._get_patch(self.pag_img, img.shape[0], img.shape[1])
            img_mask = img / 255
            img_mask = np.repeat(img_mask[:,:,np.newaxis], 3, axis=-1)
            img = patch_pag * img_mask + patch_ink * np.abs(img_mask - 1)
            img = img.astype(np.uint8)
        return img
    
def unpad(arr, value=255):
    res = arr == value
    h_proj = np.invert(res.all(1))
    w_proj = np.invert(res.all(0))
    x0, x1 = h_proj.argmax(), len(h_proj) - np.flip(h_proj).argmax()
    y0, y1 = w_proj.argmax(), len(w_proj) - np.flip(w_proj).argmax()
    return arr[x0:x1+1, y0:y1+1]

if __name__ == '__main__':
    src_dir = 'fonts'
    word_count = 75000
    sampled_words = sample(words.words(), word_count)
    start_t = time.time()
    for filename in os.listdir(src_dir):
        font_path = os.path.join(src_dir, filename)
        os.makedirs(f'dataset/fonts/{filename}', exist_ok=True)

        render = Render(font_path, calibrate=True)
        render.set_templates('blank_page.png', 'black_page.png')
        for i, word in enumerate(sampled_words):
            img = render.render(word)
            img.save(f'dataset/fonts/{filename}/{i:03d}_{word}.jpg')
            print(f'  {filename} [{i+1}/{len(sampled_words)}] {word}'.ljust(60), end='\r')
        print(f'  {filename} [{i+1}/{len(sampled_words)}] completed in {time.time() - start_t:.02} s'.ljust(60))
        break

