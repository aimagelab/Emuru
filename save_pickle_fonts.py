import os
import pickle
from pathlib import Path

fonts_path = Path('files/font_square/clean_fonts')

# Get a list of all font files in the directory
font_files = sorted(list(fonts_path.glob('*.?tf')))

# Read each font file as binary data and store the data in a dictionary
fonts_data = {}
for font_file in font_files:
    with open(font_file, 'rb') as f:
        fonts_data[font_file] = f.read()

# Save the fonts data into a pickle file
with open('files/font_square/fonts_data.pkl', 'wb') as f:
    pickle.dump(fonts_data, f)