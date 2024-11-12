from emuru import Emuru
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from torchvision.transforms import functional as F
from torchvision.utils import save_image
import argparse
import torch
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from einops import rearrange, repeat
import random
import numpy as np
from custom_datasets import dataset_factory, OnlineFontSquare
import json
from hwd.datasets.shtg import IAMLines, IAMWordsFromLines, CVLLines, RimesLines, KaraokeLines

from torch.utils.data import Dataset
from torchvision import transforms as T
from .generate_iam_emuru import SHTGWrapper, trim_white

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='files/checkpoints/Emuru_large_100k_vae2_tune2')
parser.add_argument('--dataset', type=str, default='iam_words')
args = parser.parse_args()

assert torch.cuda.device_count() == 2

dst_root = Path('files/evaluation') / (Path(args.checkpoint).name + '_' + args.dataset)
model = Emuru(t5_checkpoint='google-t5/t5-large')

checkpoint_dir = Path(args.checkpoint)
checkpoint_path = sorted(Path(checkpoint_dir).rglob('*.pth'))[-1]
print(f'Using checkpoint {checkpoint_path}')
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model'], strict=False)
model.eval()

model.T5.to('cuda:0')
model.vae.to('cuda:1')

if args.dataset == 'iam_lines': 
    dataset = IAMLines(num_style_samples=1, load_gen_sample=True)
elif args.dataset == 'iam_words': 
    dataset = IAMWordsFromLines(num_style_samples=1, load_gen_sample=True)
elif args.dataset == 'cvl_lines': 
    dataset = CVLLines(num_style_samples=1, load_gen_sample=True)
elif args.dataset == 'rimes_lines':
    dataset = RimesLines(num_style_samples=1, load_gen_sample=True)
elif args.dataset == 'karaoke_handw_lines':
    dataset = KaraokeLines('handwritten', num_style_samples=1, load_gen_sample=True)
elif args.dataset == 'karaoke_typew_lines':
    dataset = KaraokeLines('typewritten', num_style_samples=1, load_gen_sample=True)
dataset = SHTGWrapper(dataset)

dst_root.mkdir(parents=True, exist_ok=True)
dataset.dataset.save_transcriptions(dst_root)

with torch.inference_mode():
    for idx, sample in enumerate(tqdm(dataset)):
        img = sample['style_img'].to(device).unsqueeze(0)
        style_text = sample['style_imgs_text'][0]
        gen_text = sample['gen_text']
        dst_path = dst_root / Path(sample['dst_path'])

        gen_w, gen_h = sample['gen_img'].size
        tgt_w = gen_w * 64 / gen_h
        tgt_tokens = int(tgt_w / 8 * 1.5)

        result = model.generate(style_text + ' ' + gen_text, img=img, max_new_tokens=tgt_tokens)

        if img.size(-1) < result.size(-1):
            result = result[..., img.size(-1):]
            result = trim_white(result)
        else:
            result = torch.ones_like(result[..., :64])

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(result, dst_path)
