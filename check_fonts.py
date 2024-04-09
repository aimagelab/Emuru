from fontTools.ttLib import TTFont
from fontTools.pens.recordingPen import RecordingPen
from fontTools.pens.freetypePen import FreeTypePen
from fontTools.misc.transform import Offset
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from hashlib import sha1
from collections import defaultdict
from tqdm import tqdm
import json

src = Path('files/font_square/fonts')
fonts_list = sorted(src.glob('*.ttf'))

data = {}

for i, font_path in tqdm(enumerate(fonts_list), total=len(fonts_list)):
    try:
        font = TTFont(str(font_path))

        charset = defaultdict(list)
        for key, glyph in font.getGlyphSet().items():
            pen = FreeTypePen(None)
            glyph.draw(pen)
            width, ascender, descender = glyph.width, font['OS/2'].usWinAscent, -font['OS/2'].usWinDescent
            height = ascender - descender
            arr = pen.array(width=width, height=height, transform=Offset(0, -descender))
            img = Image.fromarray(np.uint8(arr * 255))

            # if width > 0 and height > 0:
            #     img.save('test.png')
            #     print(f'{key}: {arr.shape}')
            # else:
            #     print(f'{key}: has no width or height')

            if not arr.flags['C_CONTIGUOUS']:
                arr = np.ascontiguousarray(arr)
            
            if np.count_nonzero(arr) == 0:
                continue
            h = sha1(arr).hexdigest()
            charset[h].append(key)

        single_chars = [v[0] for v in charset.values() if len(v) == 1]
        data[font_path.name] = {'status': 'success', 'charset': single_chars}
    except Exception as e:
        data[font_path.name] = {'error': str(e), 'status': 'failed'}

    if i % 10 == 0:
        with open('files/check_fonts.json', 'w') as f:
            json.dump(data, f, indent=4)
with open('files/check_fonts.json', 'w') as f:
    json.dump(data, f, indent=4)