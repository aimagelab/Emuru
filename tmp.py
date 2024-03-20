from datasets import OnlineFontSquare, TextSampler
from pathlib import Path
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from torch.utils.data import DataLoader

dataset = OnlineFontSquare('files/font_square/fonts', 'files/font_square/backgrounds', TextSampler(8, 32, 6))
loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=dataset.collate_fn)

for imgs, bw_imgs, texts in tqdm(loader):
    save_image(make_grid(imgs, nrow=4), 'test.png')
    save_image(make_grid(bw_imgs, nrow=4), 'test_bw.png')
    break