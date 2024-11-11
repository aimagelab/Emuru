from custom_datasets import OnlineFontSquare
from pathlib import Path
from torchvision.utils import save_image
from tqdm import tqdm
import json
import random

class LyricsSampler:
    def __init__(self, lyrics, split_words=False):
        self.counter = 0
        self.lyrics = lyrics
        self.charset = set(' !"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~\x80\x91\x92\x93\x94\x97\x99¡¢¦§¨®°´µº»ÀÁÂÄÅÉÓÖÜßàáâãäåçèéêëíîïñóôõöùúûüĂĄčďĺŁŕŚŤťżƒ˘˝—“”╜')
        if split_words:
            self.lyrics = [word for line in self.lyrics for word in line.split()]
        self.lyrics = [t for t in self.lyrics if len(t) > 2]
    
    def __len__(self):
        return len(self.lyrics)

    def __call__(self):
        text = self.lyrics[self.counter % len(self.lyrics)]
        self.counter += 1
        text = ''.join(c for c in text if c in self.charset)
        return text
    
songs = Path('files/songs')
lyrics = [line for song in songs.glob('*.txt') for line in song.read_text().splitlines()]
lyrics = set(lyrics)
lyrics.remove('')
lyrics = list(lyrics)

def gen(sampler, fonts_path, imgs_path):
    fonts = list(Path(fonts_path).rglob('*.ttf'))
    fonts = {f.parent.name:f for f in fonts if f.parent.name != 'static'}
    dataset = OnlineFontSquare(list(fonts.values()), None, sampler)
    dataset.transform.transforms.pop(11)
    dataset.transform.transforms.pop(9)
    dataset.transform.transforms.pop(7)
    dataset.transform.transforms.pop(2)
    dataset.transform.transforms.pop(1)

    transcriptions = {}
    dst = Path(imgs_path)
    for i in tqdm(range(len(sampler))):
        db_i = i % len(dataset)
        sample = dataset[db_i]
        dataset.fonts[db_i].stem
        tmp_dst = dst / dataset.fonts[db_i].stem / f'{i:03d}.png'
        tmp_dst.parent.mkdir(parents=True, exist_ok=True)
        save_image(sample['text_img'], tmp_dst)

        transcriptions[str(tmp_dst.relative_to(dst))] = sample['text']
    with open(dst / 'transcriptions.json', 'w') as f:
        json.dump(transcriptions, f)
    

gen(LyricsSampler(lyrics), 'files/handwritten_fonts', 'files/handwritten_images/lines')
gen(LyricsSampler(lyrics), 'files/typewritten_fonts', 'files/typewritten_images/lines')
gen(LyricsSampler(lyrics, split_words=True), 'files/handwritten_fonts', 'files/handwritten_images/words')
gen(LyricsSampler(lyrics, split_words=True), 'files/typewritten_fonts', 'files/typewritten_images/words')