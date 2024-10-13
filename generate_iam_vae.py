from emuru import Emuru
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from torchvision.transforms import functional as F
from torchvision.utils import save_image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--src_root', type=str, default='files/evaluation/iam_lines')
args = parser.parse_args()

device = torch.device('cuda')
src_root = Path(args.src_root)
dst_root = src_root.parent / f'{src_root.name}_vae'
model = Emuru()

# checkpoint_dir = Path('files/checkpoints/Emuru_100k_FW')
# checkpoint_path = sorted(Path(checkpoint_dir).rglob('*.pth'))[-1]
# checkpoint = torch.load(checkpoint_path, map_location=device)
# model.load_state_dict(checkpoint['model'], strict=False)
# model.eval().to(device)
model.to(device)

images = list(src_root.rglob('*.png'))
for img_path in tqdm(images):
    pil_img = Image.open(img_path).convert('RGB')
    pil_img = pil_img.resize((pil_img.width * 64 // pil_img.height, 64))
    if pil_img.width < 16:
        continue
    img = F.to_tensor(pil_img)
    img = F.normalize(img, (0.5,), (0.5,))
    img = img.to(device).unsqueeze(0)

    decoder_inputs_embeds, z_sequence, z = model._img_encode(img)
    vae_img = model.vae.decode(z).sample

    dst_img = dst_root / img_path.relative_to(src_root)
    dst_img.parent.mkdir(parents=True, exist_ok=True)
    save_image(vae_img, dst_img)
