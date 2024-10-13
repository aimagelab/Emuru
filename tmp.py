import pickle
from pathlib import Path
from tqdm import tqdm

dst_root = Path('files/evaluation/iam_lines')
train = Path('files/pkl_db/iam_lines_test.pkl')
iam_words = Path('files/evaluation/iam')
authors = set([p.stem for p in iam_words.iterdir() if p.is_dir()])

with open(train, 'rb') as f:
    data = pickle.load(f)

for img_path, img in tqdm(zip(data['imgs'], data['imgs_preloaded'])):
    author = data['imgs_to_author'][img_path.stem]
    if author in authors:
        dst = dst_root / author / img_path.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst)