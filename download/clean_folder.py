from pathlib import Path
import shutil
from tqdm import tqdm
from collections import defaultdict

# src = Path('/home/vpippi/Emuru/files/font_square/ext_fonts')
# dst = Path('/home/vpippi/Emuru/files/font_square/clean_fonts')
# dst.mkdir(parents=True, exist_ok=True)

# exts = ['ttf', 'otf', 'woff', 'woff2']

# for file in tqdm(src.rglob('*')):
#     if file.is_file() and file.suffix[1:].lower() not in exts:
#         file.unlink()
#     elif file.is_file():
#         dst_file = dst / file.name
#         if not dst_file.exists():
#             try:
#                 shutil.copy(file, dst / file.name)
#             except Exception as e:
#                 print(f'Error copying {file}: {e}')

src = Path('/home/vpippi/Emuru/files/font_square/clean_fonts')

exts = ['ttf', 'otf']

# for file in tqdm(src.rglob('*')):
#     if file.is_file() and file.suffix[1:].lower() not in exts:
#         file.unlink()

fonts = defaultdict(list)

for file in tqdm(src.rglob('*')):
    if file.is_file():
        fonts[file.stem].append(file)

for key, value in tqdm(fonts.items()):
    if len(value) > 1:
        for file in value[:-1]:
            file.unlink()

