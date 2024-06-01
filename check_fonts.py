from fontTools.ttLib import TTFont
from fontTools.pens.freetypePen import FreeTypePen
from fontTools.misc.transform import Offset
from pathlib import Path
import numpy as np
from hashlib import sha1
from collections import defaultdict
from tqdm import tqdm
import json
import multiprocessing
from PIL import Image
import matplotlib.pyplot as plt
from custom_datasets.font_square.render_font import Render
import cv2
import string

FONT_SQUARE_CHARSET = [' ', '!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2',
                       '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E',
                       'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
                       'Y', 'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
                       'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~',
                       '\x80', '\x91', '\x92', '\x93', '\x94', '\x97', '\x99', '¡', '¢', '¦', '§', '¨', '®', '°', '´',
                       'µ', 'º', '»', 'À', 'Á', 'Â', 'Ä', 'Å', 'É', 'Ó', 'Ö', 'Ü', 'ß', 'à', 'á', 'â', 'ã', 'ä', 'å',
                       'ç', 'è', 'é', 'ê', 'ë', 'í', 'î', 'ï', 'ñ', 'ó', 'ô', 'õ', 'ö', 'ù', 'ú', 'û', 'ü', 'Ă', 'Ą',
                       'č', 'ď', 'ĺ', 'Ł', 'ŕ', 'Ś', 'Ť', 'ť', 'ż', 'ƒ', '˘', '˝', '—', '“', '”', '╜']


def char_in_font(unicode_char, font):
    '''check if a char is in font, return its glyphName'''
    for cmap in font['cmap'].tables:
        if cmap.isUnicode() or cmap.getEncoding() == 'utf_16_be':
            if ord(unicode_char) in cmap.cmap:
                # print(type(cmap))
                auxcn = cmap.cmap[ord(unicode_char)]
                # print(auxcn, type(auxcn))
                return auxcn if auxcn != '' else '<nil>'
    return ''


def create_image_grid(image_list, grid_shape=None, figsize=(10, 10), cmap='gray', save_path=None):
    """
    Creates a grid of images from a list of numpy arrays and optionally saves it as a PNG file.

    Parameters:
        image_list (list of numpy arrays): List of black and white images.
        grid_shape (tuple): Shape of the grid in (rows, columns). If None, it will be calculated automatically.
        figsize (tuple): Size of the entire figure.
        cmap (str): Color map to use for displaying images.
        save_path (str): File path to save the image grid as a PNG file. If None, the grid will not be saved.

    Returns:
        None
    """
    num_images = len(image_list)
    
    if grid_shape is None:
        num_cols = int(np.ceil(np.sqrt(num_images)))
        num_rows = int(np.ceil(num_images / num_cols))
    else:
        num_rows, num_cols = grid_shape
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            ax.imshow(image_list[i], cmap=cmap)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')  # Disable empty subplots
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    else:
        plt.show()

# Function to process a single font
def check_font(font_path):
    try:
        font = TTFont(str(font_path))

        charset = defaultdict(list)
        for char in FONT_SQUARE_CHARSET:
            key = char_in_font(char, font)
            if key == '':
                continue

            glyph = font.getGlyphSet().get(key)
            pen = FreeTypePen(None)
            glyph.draw(pen)
            width, ascender, descender = glyph.width, font['OS/2'].usWinAscent, -font['OS/2'].usWinDescent
            height = ascender - descender
            arr = pen.array(width=width, height=height, transform=Offset(0, -descender))
            arr = arr.astype(np.uint8)

            if not arr.flags['C_CONTIGUOUS']:
                arr = np.ascontiguousarray(arr)
            
            if char != ' ' and np.count_nonzero(arr) == 0:
                continue
            if char == ' ' and np.count_nonzero(arr) != 0:
                raise ValueError(f'Character "{char}" is not empty in font {font_path.name}')
            h = sha1(arr).hexdigest()
            charset[h].append(char)

        single_chars = [v[0] for v in charset.values() if len(v) == 1]
        # assert set(string.ascii_lowercase) - set(single_chars) == set(), f'Lowercase characters missing in font {font_path.name}'
        return font_path.name, single_chars
    except Exception as e:
        print(f'Error check characters: {font_path.name} - {str(e)}')
        return font_path.name, None
    

# Function to process a single font
def calibrate_font(font_path):
    try:
        render = Render(font_path)
        return font_path.name, render.calibrate(threshold=0.8, height=64)
    except Exception as e:
        print(f'Error calibration: {font_path.name} - {str(e)}')
        font_path.unlink()
        return font_path.name, None
    

def draw_character(font_path, char, out_height=64):
    try:
        font = TTFont(str(font_path))
        # table = font.getGlyphSet().glyfTable
        glyph_name = char_in_font(char, font)
        assert glyph_name != '', f'Character {char} not found in font {font_path.name}'
        glyph = font.getGlyphSet().get(char)
        pen = FreeTypePen(None)
        glyph.draw(pen)
        width, ascender, descender = glyph.width, font['OS/2'].usWinAscent, -font['OS/2'].usWinDescent
        height = ascender - descender
        arr = pen.array(width=width, height=height, transform=Offset(0, -descender))
        arr = arr.astype(np.uint8)
        img = Image.fromarray(arr)
        out_width = out_height * img.width // img.height
        img = img.resize((out_width, out_height))
        return np.array(img)
    except Exception as e:
        return np.zeros((out_height, out_height), dtype=np.uint8)

# Function to process fonts in parallel
def multiprocess(fonts_list, function, use_multiprocessing=True):
    results = {}
    if use_multiprocessing:
        with multiprocessing.Pool() as pool:
            for font, data in tqdm(pool.imap_unordered(function, fonts_list), total=len(fonts_list)):
                results[font] = data
    else:
        for font in tqdm(fonts_list, total=len(fonts_list)):
            results[font] = function(font)
    return results

if __name__ == "__main__":
    src = Path('files/font_square/clean_fonts')
    fonts_list = sorted(src.glob('*.?tf'))

    # characters = []
    # for font in fonts_list:
    #     # print(font)
    #     # arr = draw_character(font, '0')
    #     # characters.append(arr)
    #     # if len(characters) == 100:
    #     #     create_image_grid(characters, grid_shape=(10, 10), figsize=(10, 10), save_path='files/characters.png')
    #     #     characters = []
    #     check_font(font)

    use_multiprocessing = True

    data = multiprocess(fonts_list, check_font, use_multiprocessing)
    with open('files/fonts_charsets.json', 'w') as f:
        json.dump(data, f)

    data = multiprocess(fonts_list, calibrate_font, use_multiprocessing)
    with open('files/fonts_sizes.json', 'w') as f:
        json.dump(data, f)
    


    

    
