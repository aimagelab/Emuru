from emuru import Emuru
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torch
from torchvision.transforms import functional as F
from torchvision.utils import save_image
import argparse
from models.autoencoder_kl import AutoencoderKL


def generate(vae_path, device, src_root, dst_root):
    vae = AutoencoderKL.from_pretrained(vae_path, subfolder="vae")
    vae = vae.eval()
    vae = vae.to(device)

    images = list(src_root.rglob('*.png'))
    for img_path in tqdm(images):
        pil_img = Image.open(img_path).convert('RGB')
        pil_img = pil_img.resize((pil_img.width * 64 // pil_img.height, 64))
        if pil_img.width < 16:
            continue
        img = F.to_tensor(pil_img)
        img = F.normalize(img, (0.5,), (0.5,))
        img = img.to(device).unsqueeze(0)

        posterior = vae.encode(img)
        z = posterior.latent_dist.sample()
        vae_img = vae.decode(z).sample

        dst_img = dst_root / img_path.relative_to(src_root)
        dst_img.parent.mkdir(parents=True, exist_ok=True)

        img = vae_img[0]

        save_image(img, dst_img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_root', type=str, default='files/evaluation/iam_lines')
    args = parser.parse_args()

    # args.src_root = 'files/evaluation/iam_lines'
    # device = torch.device('cuda')
    # src_root = Path(args.src_root)
    # dst_root = src_root.parent / f'{src_root.name}_vae_iaml'
    # vae_path = '/home/fquattrini/emuru/results_vae_iaml/3278/model_0353'
    # generate(vae_path, device, src_root, dst_root)

    # args.src_root = 'files/evaluation/iam_words'
    # device = torch.device('cuda')
    # src_root = Path(args.src_root)
    # dst_root = src_root.parent / f'{src_root.name}_vae_iaml'
    # vae_path = '/home/fquattrini/emuru/results_vae_iaml/3278/model_0353'
    # generate(vae_path, device, src_root, dst_root)

    # args.src_root = 'files/evaluation/iam_lines'
    # device = torch.device('cuda')
    # src_root = Path(args.src_root)
    # dst_root = src_root.parent / f'{src_root.name}_vae_iaml_finetune'
    # vae_path = '/home/fquattrini/emuru/results_vae_iaml_finetune/08c0/model_0373'
    # generate(vae_path, device, src_root, dst_root)

    # args.src_root = 'files/evaluation/iam_words'
    # device = torch.device('cuda')
    # src_root = Path(args.src_root)
    # dst_root = src_root.parent / f'{src_root.name}_vae_iaml_finetune'
    # vae_path ='/home/fquattrini/emuru/results_vae_iaml_finetune/08c0/model_0373'
    # generate(vae_path, device, src_root, dst_root)

    args.src_root = 'files/evaluation/iam_words'
    device = torch.device('cuda')
    src_root = Path(args.src_root)
    dst_root = src_root.parent / f'{src_root.name}_vae_diffusionpen_iaml'
    vae_path ='sd-legacy/stable-diffusion-v1-5'
    generate(vae_path, device, src_root, dst_root)

    args.src_root = 'files/evaluation/iam_lines'
    device = torch.device('cuda')
    src_root = Path(args.src_root)
    dst_root = src_root.parent / f'{src_root.name}_vae_diffusionpen_iaml'
    vae_path ='sd-legacy/stable-diffusion-v1-5'
    generate(vae_path, device, src_root, dst_root)


if __name__ == '__main__':
    main()

