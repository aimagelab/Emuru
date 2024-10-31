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


def plot_tsne(vectors, labels, filename='tsne_plot.png'):
    """
    Generates a t-SNE plot for a list of vectors with class labels and saves the plot as an image.
    
    Parameters:
        vectors (torch.Tensor): A tensor of shape (num_samples, num_features) representing the vectors.
        labels (list or array-like): A list or array of labels corresponding to the class of each vector.
        filename (str): The filename to save the plot.
    """
    # Convert to numpy for compatibility with scikit-learn
    vectors_np = vectors.cpu().numpy() if isinstance(vectors, torch.Tensor) else vectors
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(vectors_np)
    
    # Plotting the results with different colors for each class
    plt.figure(figsize=(10, 7))
    for class_id in set(labels):  # Iterate over each unique class
        plt.scatter(
            tsne_results[[class_id == l for l in labels], 0],
            tsne_results[[class_id == l for l in labels], 1],
            s=10,
            label=f'Class {class_id}'
        )
        
    plt.title("t-SNE of Vectors with Class Coloring")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend()
    
    # Save the plot
    plt.savefig(filename)
    plt.close()  # Close the plot to free memory
    
    print(f"t-SNE plot saved as '{filename}'")

def plot_1d_array(data, title="1D Array Plot", filename='token_sim.png'):
    """
    Plots a simple 1D array and optionally saves the plot as an image.
    
    Parameters:
        data (list or array-like): The 1D data to plot.
        title (str): Title of the plot.
        filename (str, optional): If provided, saves the plot to this file.
    """
    plt.figure()
    plt.plot(data, marker='o')
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Value")
    
    plt.savefig(filename)
    print(f"Plot saved as '{filename}'")
    plt.close()  # Close the plot to free memory

def plot_1d_array(data):
    """
    Plots a simple 1D array and optionally saves the plot as an image.
    
    Parameters:
        data (list or array-like): The 1D data to plot.
        title (str): Title of the plot.
        filename (str, optional): If provided, saves the plot to this file.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data.cpu().numpy(), bins=30, alpha=0.7, color='blue', edgecolor='black')
    plt.title('Distribution of 1D Tensor Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Save the plot as a figure
    plt.savefig('distribution_plot.png')  # Save as PNG file
    plt.close()  # Close the plot


parser = argparse.ArgumentParser()
parser.add_argument('--src_root', type=str, default='files/evaluation/iam_lines')
args = parser.parse_args()

device = torch.device('cuda')
src_root = Path(args.src_root)
dst_root = src_root.parent / f'emuru_vae2'
model = Emuru()

checkpoint_dir = Path('files/checkpoints/Emuru_100k_vae2')
checkpoint_path = sorted(Path(checkpoint_dir).rglob('*.pth'))[-1]
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model'], strict=False)
model.eval().to(device)
# model.to(device)

images = list(src_root.rglob('*.png'))
random.shuffle(images)

learning_steps = 100
text_tokens = []
pad_tokens = []
with torch.inference_mode():
    for idx, img_path in enumerate(tqdm(images)):
        pil_img = Image.open(img_path).convert('RGB')
        pil_img = pil_img.resize((pil_img.width * 64 // pil_img.height, 64))
        if pil_img.width < 16:
            continue
        img = F.to_tensor(pil_img)

        c, h, w = img.shape
        canvas = torch.ones((c, h, w + 8 * 16))
        canvas[:, :, :w] = img
        img = canvas

        img = F.normalize(img, (0.5,), (0.5,))
        img = img.to(device).unsqueeze(0)

        decoder_inputs_embeds, z_sequence, z = model._img_encode(img)

        if idx < learning_steps:
            tokens = z.squeeze().T.cpu()
            text_tokens.append(tokens[:-8])
            pad_tokens.append(tokens[-8:])
        elif idx == learning_steps:
            text_tokens = torch.cat(text_tokens)
            pad_tokens = torch.cat(pad_tokens)
            avg_pad = pad_tokens.mean(0).unsqueeze(0).to(z.device)
            sim = torch.nn.functional.cosine_similarity(pad_tokens, avg_pad.cpu())
            threshold = np.percentile(sim.numpy(), 1)
        elif idx > learning_steps:
            vae_img = model.vae.decode(z).sample
            sim = torch.nn.functional.cosine_similarity(z.squeeze().T, avg_pad)

            pad_idx = torch.where(sim > threshold)[0]
            vae_img = torch.cat(vae_img.split(8, dim=-1))
            vae_img[pad_idx, :, 0::2, 1::2] = 0
            save_image(vae_img, f'test_find_pad_{idx % 10}.png', nrow=vae_img.size(0))

            # dst_img = dst_root / img_path.relative_to(src_root)
            # dst_img.parent.mkdir(parents=True, exist_ok=True)

            # img = vae_img[0]

            # # Create a white background (shape matches RGB, value 1 for white)
            # white_bg = torch.ones_like(rgb)

            # # Blend the RGB with the white background using the alpha channel
            # vae_img = rgb * alpha + white_bg * (1 - alpha)
            # save_image(img, dst_img)


# text_tokens = torch.cat(text_tokens)
# pad_tokens = torch.cat(pad_tokens)
# vectors = torch.cat([text_tokens, pad_tokens])
# labels = ['text'] * len(text_tokens) + ['pad'] * len(pad_tokens)
# labels[:len(text_tokens)] = 1

# plot_tsne(vectors, labels, filename='tsne_plot.png')
