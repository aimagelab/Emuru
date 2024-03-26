from datasets import OnlineFontSquare, TextSampler
from pathlib import Path
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from torch.utils.data import DataLoader
import string
import random
import torch

class FakeTextSampler:
    def __init__(self):
        text_sampler = TextSampler(8, 32, 6)
        self.corpus = set(''.join(text_sampler.words)) - set(string.ascii_letters + string.digits + ' ')
        self.corpus = ''.join(sorted(self.corpus))

    def __call__(self):
        return self.corpus

random.seed(0)
torch.manual_seed(0)

dataset = OnlineFontSquare('files/font_square/fonts', 'files/font_square/backgrounds', TextSampler(8, 32, 6))
loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn, num_workers=0)

for i, batch in tqdm(enumerate(loader)):
    # save_image(make_grid(imgs, nrow=4), 'test.png')
    # save_image(make_grid(bw_imgs, nrow=4), 'test_bw.png')
    if i >= 20:
        break
dataset.transform.print_times()

print('done')