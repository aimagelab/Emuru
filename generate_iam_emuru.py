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


def save_images_in_grid(images, file_path, nrow=4, padding=2, pad_color=0):
    """
    Saves a left-aligned grid of images with varying widths.

    Parameters:
    - images (list of torch.Tensor): List of image tensors, each of shape [C, H, W].
    - file_path (str): Path to save the grid image.
    - nrow (int): Number of images per row in the grid.
    - padding (int): Amount of padding between images.
    - pad_color (float): Padding color value, 0 for black and 1 for white.
    """
    # Find the maximum width and height among all images
    max_height = max(image.shape[1] for image in images)
    max_width = max(image.shape[2] for image in images)
    
    # Pad each image to the same width and height, aligning to the left
    padded_images = []
    for image in images:
        height, width = image.shape[1], image.shape[2]
        pad_right = max_width - width
        pad_bottom = max_height - height
        # Apply padding to the right and bottom to align left and top
        padded_image = F.pad(
            image, (0, 0, pad_right, pad_bottom), fill=pad_color
        )
        padded_images.append(padded_image)
    
    # Stack the images and save them as a grid
    grid = torch.stack(padded_images)
    save_image(grid, file_path, nrow=nrow, padding=padding)


parser = argparse.ArgumentParser()
parser.add_argument('--src_root', type=str, default='files/evaluation/iam_lines')
args = parser.parse_args()

device = torch.device('cuda')
src_root = Path(args.src_root)
dst_root = src_root.parent / f'emuru_long_iam_lines_no_eval'
model = Emuru()

# checkpoint_dir = Path('files/checkpoints/Emuru_100k_vae2')
checkpoint_dir = Path('files/checkpoints/Emuru_100k_vae2_long_iam')
checkpoint_path = sorted(Path(checkpoint_dir).rglob('*.pth'))[-1]
print(f'Using checkpoint {checkpoint_path}')
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model'], strict=False)
# model.eval().to(device)
model.to(device)

# dataset = dataset_factory('test', ['iam_eval'], root_path='/home/vpippi/Teddy/files/datasets/')
dataset = dataset_factory('test', ['iam_lines'], root_path='/home/vpippi/Teddy/files/datasets/')
dataset.batch_keys('same', 'style')
# dataset = OnlineFontSquare('files/font_square/clean_fonts', 'files/font_square/backgrounds', lambda: 'This is a sample text very long.')

learning_steps = 100
images = []
text_data = {}
with torch.inference_mode():
    for idx, sample in enumerate(tqdm(dataset)):
        # img = sample['img'].to(device).unsqueeze(0)
        # style_text = sample['text']

        # if you use the dataset iam_eval
        # img = sample['style_img'].to(device).unsqueeze(0)
        # style_text = sample['style_text']
        # gen_text = sample['gen_text']
        # dst_path = Path(sample['dst_path'])

        # if you use the datset iam_lines
        img = sample['same_img'].to(device).unsqueeze(0)
        style_text = sample['same_text']
        gen_text = sample['style_text']
        dst_path = Path('test') / sample['same_author'] / f'{idx}.png'

        text_data[str(dst_path.relative_to('test'))] = gen_text

        # decoder_inputs_embeds, z_sequence, z = model._img_encode(img)
        # result = model.generate(f'{style_text} {gen_text}', z_sequence=z_sequence, max_new_tokens=64)
        decoder_inputs_embeds, z_sequence, z = model._img_encode(img)
        result = model.generate(style_text + ' ' + gen_text, z_sequence=z_sequence[:, :-8], max_new_tokens=128)
        # print()
        if img.size(-1) < result.size(-1):
            result = result[..., img.size(-1):]
        else:
            result = torch.ones_like(result[..., :64])
        # if output.numel() == 0:
        #     print('Error empty image, saving white image instead')
        #     output = torch.ones_like(result[0, :, :, :64])
        dst_img = dst_root / dst_path
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        save_image(result, dst_img)

        # images.append(result[0])
        # if len(images) == 16:
        #     dst_img = dst_root / f'{idx}.png'
        #     dst_img.parent.mkdir(parents=True, exist_ok=True)
        #     save_images_in_grid(images, dst_img, nrow=1)
        #     images = []

with open(dst_root / 'transcriptions.json', 'w') as f:
    json.dump(text_data, f, indent=2)

# text_tokens = torch.cat(text_tokens)
# pad_tokens = torch.cat(pad_tokens)
# vectors = torch.cat([text_tokens, pad_tokens])
# labels = ['text'] * len(text_tokens) + ['pad'] * len(pad_tokens)
# labels[:len(text_tokens)] = 1

# plot_tsne(vectors, labels, filename='tsne_plot.png')
