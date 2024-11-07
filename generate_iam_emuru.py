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
from hwd.datasets.shtg import IAMLines, IAMWords, CVLLines, Rimes

from torch.utils.data import Dataset
from torchvision import transforms as T

class SHTGWrapper(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.transforms = T.Compose([
            self._to_height_64,
            T.ToTensor()
        ])

    def _to_height_64(self, img):
        width, height = img.size
        aspect_ratio = width / height
        new_width = int(64 * aspect_ratio)
        
        # Resize the image
        resized_image = img.resize((new_width, 64), Image.LANCZOS)
        return resized_image


    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        sample = self.dataset[index]
        sample['style_img'] = self.transforms(sample['style_imgs'][0].convert('RGB'))
        return sample

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='files/checkpoints/Emuru_large_100k_vae2_tune')
parser.add_argument('--dataset', type=str, default='iam_lines')
parser.add_argument('--dst', type=str, default='files/evaluation/emuru_large_tune_iam_lines')
args = parser.parse_args()

device = torch.device('cuda')
dst_root = Path(args.dst)
model = Emuru(t5_checkpoint='google-t5/t5-large')

checkpoint_dir = Path(args.checkpoint)
checkpoint_path = sorted(Path(checkpoint_dir).rglob('*.pth'))[-1]
print(f'Using checkpoint {checkpoint_path}')
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model'], strict=False)
model.eval().to(device)

if args.dataset == 'iam_lines': 
    dataset = IAMLines(num_style_samples=1, scenario='test')
elif args.dataset == 'iam_words': 
    raise NotImplementedError
    dataset = IAMLines(num_style_samples=1, scenario='test')
elif args.dataset == 'cvl': 
    dataset = CVLLines(num_style_samples=1, scenario='test')
elif args.dataset == 'rimes':
    dataset = Rimes(num_style_samples=1, scenario='test')
dataset = SHTGWrapper(dataset)

text_data = {}
with torch.inference_mode():
    for idx, sample in enumerate(tqdm(dataset)):
        # img = sample['img'].to(device).unsqueeze(0)
        # style_text = sample['text']

        # if you use the dataset iam_eval
        img = sample['style_img'].to(device).unsqueeze(0)
        style_text = sample['style_imgs_text'][0]
        gen_text = sample['gen_text']
        dst_path = dst_root / Path(sample['dst_path']).relative_to('test')

        # # if you use the datset iam_lines
        # img = sample['same_img'].to(device).unsqueeze(0)
        # style_text = sample['same_text']
        # gen_text = sample['style_text']
        # dst_path = Path('test') / sample['same_author'] / f'{idx}.png'

        text_data[str(Path(sample['dst_path']).relative_to('test'))] = gen_text

        # decoder_inputs_embeds, z_sequence, z = model._img_encode(img)
        # result = model.generate(f'{style_text} {gen_text}', z_sequence=z_sequence, max_new_tokens=64)
        decoder_inputs_embeds, z_sequence, z = model._img_encode(img)
        result = model.generate(style_text + ' ' + gen_text, z_sequence=z_sequence[:, :-8], max_new_tokens=128)

        if img.size(-1) < result.size(-1):
            result = result[..., img.size(-1):]
        else:
            result = torch.ones_like(result[..., :64])

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(result, dst_path)

with open(dst_root / 'transcriptions.json', 'w') as f:
    json.dump(text_data, f, indent=2)