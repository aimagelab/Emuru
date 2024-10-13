from custom_datasets.font_square.font_square import make_renderers, get_fonts
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import json
from concurrent.futures import ProcessPoolExecutor
from matplotlib import pyplot as plt
import numpy as np
import torch
from torchvision.utils import save_image, make_grid

fonts = get_fonts(Path('/home/vpippi/Emuru/files/font_square/clean_fonts'))
renderers = make_renderers(fonts, calib_threshold=0.8, verbose=True, load_font_into_mem=False)

charset = [r.charset for r in renderers if r.charset is not None]
charset = set().union(*charset)
for renderer in renderers:
    renderer.charset = None

class NumpyChar:
    def __init__(self, char, np_img):
        self.char = char
        self.np_img = np_img

    def __eq__(self, other):
        return np.array_equal(self.np_img, other.np_img)

    def __repr__(self):
        return self.char
    
    def tensor(self):
        return torch.tensor(self.np_img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

def save_grid(numpy_chars, path, ncols=16):
    nrows = len(numpy_chars) // ncols + 1
    f, axarr = plt.subplots(nrows, ncols, figsize=(ncols, nrows))
    axarr = axarr.flatten()
    for idx, numpy_char in enumerate(numpy_chars):
        axarr[idx].imshow(numpy_char.np_img, cmap='gray')
        axarr[idx].axis('off')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def check_font(in_charset, renderer):
    numpy_chars = []
    allowed_chars = []
    removed_chars = []
    for char in sorted(in_charset):
        try:
            np_img, _ = renderer.render(char, action='top_left', pad=0)
            assert (np_img == 0).sum() > 0 or char == ' '

            numpy_char = NumpyChar(char, np_img)
            numpy_chars.append(numpy_char)

            if numpy_char in removed_chars:
                continue
            if numpy_char in allowed_chars:
                allowed_chars.remove(numpy_char)
                removed_chars.append(numpy_char)
            else:
                allowed_chars.append(numpy_char)
        except Exception as e:
            continue

    # numpy_chars[0] == numpy_chars[1]
    # save_grid(allowed_chars, f'allowed.png')
    # save_grid(removed_chars, f'removed.png')
    # save_grid(numpy_chars, f'all.png')
    out_charset = [c.char for c in allowed_chars]
    return out_charset, Path(renderer.font_path).name

# for idx, char in enumerate(sorted(charset)):
#     for renderer in tqdm(renderers, desc=f'"{char}" - [{idx + 1}/{len(charset)}]'):
#         try:
#             np_img, out_char = renderer.render(char, action='top_left', pad=0)
#             assert (np_img == 0).sum() > 0 or char == ' '
#             verified_charset[Path(renderer.font_path).name].append(char)
#         except Exception as e:
#             continue
import time

if Path('verified_charset.json').exists():
    with open('verified_charset.json', 'r') as f:
        verified_charset = json.load(f)
else:
    verified_charset = {}

    # check_font(charset, renderers[1001])

    with ProcessPoolExecutor() as executor:
        futures = []
        for renderer in tqdm(renderers, desc='Submitting fonts'):
            futures.append(executor.submit(check_font, charset, renderer))
            time.sleep(0.0001)
        for future in tqdm(futures, desc='Checking fonts'):
            out_charset, font_key = future.result()
            verified_charset[font_key] = out_charset

    print('Saving verified_charset.json')
    with open('verified_charset.json', 'w') as f:
        json.dump(verified_charset, f)

comp = {c for l in tqdm(verified_charset.values()) for c in l}
print('Done')

count = np.array([len(val) for val in verified_charset.values()])

plt.hist(count, bins=50)
plt.savefig('hist.png')