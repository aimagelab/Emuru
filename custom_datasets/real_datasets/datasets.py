import itertools
import random
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image
import torch
import html
import gzip
import pickle
from torch.utils.data import Dataset
import msgpack
import json
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
from einops import rearrange
from . import transforms as T
from itertools import compress


def get_alphabet(labels):
    coll = ''.join(labels)
    unq = sorted(list(set(coll)))
    unq = [''.join(i) for i in itertools.product(unq, repeat=1)]
    alph = dict(zip(unq, range(len(unq))))
    return alph


def pad_images(images, padding_value=1):
    images = [rearrange(img, 'c h w -> w c h') for img in images]
    return rearrange(pad_sequence(images, padding_value=padding_value), 'w b c h -> b c h w')


class Base_dataset(Dataset):
    def __init__(self, path, nameset='train', transform=T.ToTensor(), pkl_path=None):
        super().__init__()
        if isinstance(transform, tuple):
            self.pre_transform, self.post_transform = transform
        else:
            self.pre_transform, self.post_transform = None, transform
        self.nameset = nameset
        self.path = path

        self.imgs_to_idx = {}
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.author_to_imgs = {}
        self.imgs_set = set()
        self.preloaded = False
        self.batch_keys = ['style', 'other', 'same']
        self.multiplier = 1

        if pkl_path is None or not pkl_path.exists():
            self.imgs = []
            self.imgs_to_label = {}
            self.imgs_to_author = {}
            self.imgs_to_sizes = {}
        else:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            self.imgs = data['imgs']
            self.imgs_preloaded = data['imgs_preloaded']
            self.imgs_to_label = data['imgs_to_label']
            self.imgs_to_author = data['imgs_to_author']
            self.imgs_to_sizes = data['imgs_to_sizes']
            self.preloaded = True

    def __len__(self):
        return len(self.imgs)
    
    def _get_sample(self, idx, sample=None, tag=None):
        path = self.imgs[idx]
        text = self.imgs_to_label[path.stem]
        img = Image.open(path) if not self.preloaded else self.imgs_preloaded[idx]

        if self.pre_transform and not self.preloaded:
            img, _ = self.pre_transform((img, text))
        if self.post_transform:
            img, _ = self.post_transform((img, text))

        img_len = img.shape[-1]
        author = self.imgs_to_author[path.stem]

        if sample is not None and tag is not None: 
            sample[f'{tag}_img'] = img
            sample[f'{tag}_img_len'] = img_len
            sample[f'{tag}_text'] = text
            sample[f'{tag}_author'] = author

        return img, img_len, text, author

    def __getitem__(self, idx):
        sample = {}
        *_, style_author = self._get_sample(idx, sample, 'style')

        if 'other' in self.batch_keys or 'same' in self.batch_keys:
            same_author_imgs = self.author_to_imgs[style_author]
            other_author_imgs = self.imgs_set - same_author_imgs
            other_author_imgs = other_author_imgs if len(other_author_imgs) > 0 else same_author_imgs

            if 'other' in self.batch_keys:
                path = random.choice(list(other_author_imgs))
                idx = self.imgs_to_idx[path]
                self._get_sample(idx, sample, 'other')

            if 'same' in self.batch_keys:
                path = random.choice(list(same_author_imgs))
                idx = self.imgs_to_idx[path]
                self._get_sample(idx, sample, 'same')

            # Snippet of code used to generate multiple same images
            if self.multiplier > 1 and 'same' in self.batch_keys:
                for i in range(self.multiplier - 1):
                    path = random.choice(list(same_author_imgs))
                    idx = self.imgs_to_idx[path]
                    self._get_sample(idx, sample, f'same_{i}')
        return sample

    def load_img_sizes(self, img_sizes_path):
        if img_sizes_path.exists():
            with open(img_sizes_path, 'rb') as f:
                img_sizes = msgpack.load(f, strict_map_key=False)
            return {Path(filename).stem: (width, height) for filename, width, height in img_sizes}
        else:
            data = []
            img_sizes = {}
            for img_path in tqdm(self.imgs, desc='Loading image sizes'):
                try:
                    img = Image.open(img_path)
                except:
                    print(f'Error opening {img_path}')
                    continue
                img_sizes[img_path.stem] = img.size
                data.append((img_path.name, *img.size))
            with open(img_sizes_path, 'wb') as f:
                msgpack.dump(data, f)
            return img_sizes

    def preload(self):
        self.imgs_preloaded = [(Image.open(img_path), self.imgs_to_label[img_path.stem]) for img_path in self.imgs]
        if self.pre_transform:
            self.imgs_preloaded = [self.pre_transform((img, lbl))[0] for img, lbl in self.imgs_preloaded]
        self.preloaded = True

    def save_pickle(self, path):
        imgs_preloaded = [(Image.open(img_path), self.imgs_to_label[img_path.stem]) for img_path in tqdm(self.imgs, desc='Reading images')]
        imgs_preloaded = [self.pre_transform((img, lbl))[0] for img, lbl in tqdm(imgs_preloaded, desc='Preprocessing images')]
        data = {
            'imgs': self.imgs,
            'imgs_preloaded': imgs_preloaded,
            'imgs_to_label': self.imgs_to_label,
            'imgs_to_author': self.imgs_to_author,
            'imgs_to_sizes': self.imgs_to_sizes,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f'Dataset saved to {path}')

class IAM_dataset(Base_dataset):
    def __init__(self, path, nameset=None, transform=T.ToTensor(), max_width=None, max_height=None, dataset_type='lines', preload=False, pkl_path=None, min_text_width=None, **kwargs):
        super().__init__(path, nameset, transform, pkl_path)
        self.dataset_type = dataset_type

        if pkl_path is None or not pkl_path.exists():
            self.imgs = list(Path(path, self.dataset_type).rglob('*.png'))

            xml_files = [ET.parse(xml_file) for xml_file in Path(path, 'xmls').rglob('*.xml')]
            tag = 'line' if self.dataset_type.startswith('lines') else 'word'
            self.imgs_to_label = {el.attrib['id']: html.unescape(el.attrib['text']) for xml_file in xml_files for el in xml_file.iter() if el.tag == tag}
            self.imgs_to_author = {el.attrib['id']: xml_file.getroot().attrib['writer-id'] for xml_file in xml_files for el in xml_file.iter() if el.tag == tag}

            img_sizes_path = Path(path, f'img_sizes_{dataset_type}.msgpack')
            self.imgs_to_sizes = self.load_img_sizes(img_sizes_path)

        if min_text_width:
            mask = [len(self.imgs_to_label[img.stem]) >= min_text_width for img in self.imgs]
            self.imgs = list(compress(self.imgs, mask))
            if self.preloaded:
                self.imgs_preloaded = list(compress(self.imgs_preloaded, mask))

        if pkl_path is not None and not pkl_path.exists():
            self.save_pickle(pkl_path)

        htg_train_authors = Path('files/gan.iam.tr_va.gt.filter27.txt').read_text().splitlines()
        htg_train_authors = sorted({line.split(',')[0] for line in htg_train_authors})

        htg_test_authors = Path('files/gan.iam.test.gt.filter27.txt').read_text().splitlines()
        htg_test_authors = sorted({line.split(',')[0] for line in htg_test_authors})

        target_authors = []
        if 'all' in nameset:
            target_authors += sorted(set(self.imgs_to_author.values()))
        if 'train' in nameset:
            target_authors += htg_train_authors
        if 'test' in nameset:
            target_authors += htg_test_authors
        if len(target_authors) == 0:
            raise ValueError(f'Unknown nameset {nameset}')
        target_authors = sorted(set(target_authors))

        if self.preloaded:
            self.imgs_preloaded = [img for path, img in zip(self.imgs, self.imgs_preloaded) if self.imgs_to_author[path.stem] in target_authors]
        self.imgs = [img for img in self.imgs if self.imgs_to_author[img.stem] in target_authors]
        self.imgs_to_author = {k: v for k, v in self.imgs_to_author.items() if v in target_authors}
        self.imgs_to_label = {k: v for k, v in self.imgs_to_label.items() if k in self.imgs_to_author}

        assert all(path.stem in self.imgs_to_label for path in self.imgs), 'Images and labels do not match'
        assert len(self.imgs) > 0, f'No images found in {path}'
        self.char_to_idx = get_alphabet(self.imgs_to_label.values())
        self.idx_to_char = dict(zip(self.char_to_idx.values(), self.char_to_idx.keys()))

        if max_width and max_height:
            # assert set(self.imgs_to_label.keys()) == set(self.imgs_to_sizes.keys())
            target_width = {filename: width * max_height / height for filename, (width, height) in self.imgs_to_sizes.items()}
            if self.preloaded:
                self.imgs_preloaded = [img for path, img in zip(self.imgs, self.imgs_preloaded) if path.stem in target_width and target_width[path.stem] <= max_width]
            self.imgs = [img for img in self.imgs if img.stem in target_width and target_width[img.stem] <= max_width]

        self.imgs_to_idx = {img: idx for idx, img in enumerate(self.imgs)}
        self.imgs_set = set(self.imgs)
        self.author_to_imgs = {author: {img for img in self.imgs if self.imgs_to_author[img.stem] == author} for author in target_authors}
        
        assert not self.preloaded or len(self.imgs) == len(self.imgs_preloaded), 'Preloaded images do not match'

class IAM_dataset_double(IAM_dataset):
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        sample['style_img'] = torch.cat([sample['style_img'], sample['same_img']], dim=-1)
        del sample['same_img']
        sample['style_text'] = sample['style_text'] + ' ' + sample['same_text']
        del sample['same_text']
        return sample

class IAM_custom_dataset(Base_dataset):
    def __init__(self, path, dataset_type, nameset=None, transform=T.ToTensor(), pkl_path=None, **kwargs):
        super().__init__(path, nameset, transform, pkl_path)

        if pkl_path is None or not pkl_path.exists():
            self.imgs = list(Path(path, dataset_type).rglob('*.png'))

            with open(Path(path, dataset_type, 'data.json')) as f:
                self.data = json.load(f)

            self.imgs_to_label = {img_id: value['text'] for img_id, value in self.data.items()}
            self.imgs_to_author = {img_id: value['auth'] for img_id, value in self.data.items()}

        if pkl_path is not None and not pkl_path.exists():
            self.save_pickle(pkl_path)

        htg_train_authors = Path('files/gan.iam.tr_va.gt.filter27.txt').read_text().splitlines()
        htg_train_authors = sorted({line.split(',')[0] for line in htg_train_authors})

        htg_test_authors = Path('files/gan.iam.test.gt.filter27.txt').read_text().splitlines()
        htg_test_authors = sorted({line.split(',')[0] for line in htg_test_authors})

        if nameset == 'train':
            target_authors = htg_train_authors
        elif nameset == 'test':
            target_authors = htg_test_authors
        else:
            raise ValueError(f'Unknown nameset {nameset}')

        if self.preloaded:
            self.imgs_preloaded = [img for path, img in zip(self.imgs, self.imgs_preloaded) if self.imgs_to_author[path.stem] in target_authors]
        self.imgs = [img for img in self.imgs if self.imgs_to_author[img.stem] in target_authors]
        # self.imgs_to_author = {k: v for k, v in self.imgs_to_author.items() if v in target_authors}
        # self.imgs_to_label = {k: v for k, v in self.imgs_to_label.items() if k in self.imgs_to_author}

        assert all(path.stem in self.imgs_to_label for path in self.imgs), 'Images and labels do not match'
        assert len(self.imgs) > 0, f'No images found in {path}'
        self.char_to_idx = get_alphabet(self.imgs_to_label.values())
        self.idx_to_char = dict(zip(self.char_to_idx.values(), self.char_to_idx.keys()))

        self.imgs_set = set(self.imgs)
        self.imgs_to_idx = {img: idx for idx, img in enumerate(self.imgs)}
        self.author_to_imgs = {author: {img for img in self.imgs if self.imgs_to_author[img.stem] == author} for author in target_authors}

        assert not self.preloaded or len(self.imgs) == len(self.imgs_preloaded), 'Preloaded images do not match'


class IAM_eval(IAM_dataset):
    def __init__(self, *args, **kwargs):
        kwargs['nameset'] = 'all'
        kwargs['max_width'] = None
        kwargs['max_height'] = None
        kwargs['pkl_path'] = None
        super().__init__(*args, **kwargs)

        # with gzip.open('files/iam_htg_setting.json.gz', 'rt', encoding='utf-8') as file:
        with gzip.open('files/iam_htg_setting_l.json.gz', 'rt', encoding='utf-8') as file:
            self.data = json.load(file)

        self.imgs_id_to_path = {img.stem: img for img in self.imgs}
        self.imgs_id_to_idx = {img.stem: idx for idx, img in enumerate(self.imgs)}

    def filter_eval_set(self, *eval_set):
        eval_set = set(eval_set)
        self.data = [el for el in self.data if Path(el['dst']).parts[0] in eval_set]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        gen_text = sample['word']

        style_id = sample['style_ids'][0].split('-')
        while '-'.join(style_id) not in self.imgs_to_label and len(style_id) > 0:
            style_id.pop(-1)
        assert len(style_id) > 0, f'No style found for {sample}'
        style_id = '-'.join(style_id)

        style_text = self.imgs_to_label[style_id]
        style_img = Image.open(self.imgs_id_to_path[style_id]) 

        if self.pre_transform:
            style_img, _ = self.pre_transform((style_img, style_text))
        if self.post_transform:
            style_img, _ = self.post_transform((style_img, style_text))

        dst_path = sample['dst']

        sample = {
            'gen_text': gen_text,
            'style_img': style_img,
            'style_img_len': style_img.shape[-1],
            'style_text': style_text,
            'style_author': self.imgs_to_author[style_id],
            'dst_path': dst_path
        }
        return sample
    

class IAM_custom_eval(IAM_custom_dataset):
    def __init__(self, *args, **kwargs):
        kwargs['nameset'] = 'test'
        kwargs['max_width'] = None
        kwargs['max_height'] = None
        kwargs['pkl_path'] = None
        super().__init__(*args, **kwargs)

        # with gzip.open('files/iam_lines_htg_setting.json.gz', 'rt', encoding='utf-8') as file:
        with gzip.open('files/iam_htg_setting.json.gz', 'rt', encoding='utf-8') as file:
            self.data = json.load(file)

        self.imgs_id_to_path = {img.stem: img for img in self.imgs}
        self.imgs_id_to_idx = {img.stem: idx for idx, img in enumerate(self.imgs)}

    def filter_eval_set(self, *eval_set):
        eval_set = set(eval_set)
        self.data = [el for el in self.data if Path(el['dst']).parts[0] in eval_set]
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        gen_text = sample['word']

        for style_id in sample['style_ids']:
            style_id = style_id.split('-')
            while '-'.join(style_id) + '-00' not in self.imgs_to_label and len(style_id) > 0:
                style_id.pop(-1)
            style_id = '-'.join(style_id) + '-00'
            if style_id in self.imgs_id_to_path:
                break

        assert style_id in self.imgs_id_to_path, f'No style found for {sample}'
        style_text = self.imgs_to_label[style_id]
        style_img = Image.open(self.imgs_id_to_path[style_id]) 

        if self.pre_transform:
            style_img, _ = self.pre_transform((style_img, style_text))
        if self.post_transform:
            style_img, _ = self.post_transform((style_img, style_text))

        dst_path = sample['dst']

        sample = {
            'gen_text': gen_text,
            'style_img': style_img,
            'style_img_len': style_img.shape[-1],
            'style_text': style_text,
            'style_author': self.imgs_to_author[style_id],
            'dst_path': dst_path
        }
        return sample


class Msgpack_dataset(Base_dataset):
    def __init__(self, path, nameset='train', transform=T.ToTensor(), max_width=None, max_height=None, preload=False, pkl_path=None, author_code='unknown', author_fn=None, **kwargs):
        super().__init__(path, nameset, transform, pkl_path)
        self.author_fn = author_fn if author_fn is not None else lambda *_: author_code

        if pkl_path is None or not pkl_path.exists():
            nameset_path = Path(path, f'{nameset}.msgpack')
            assert nameset_path.exists(), f'No msgpack file found in {path}'

            with open(nameset_path, 'rb') as f:
                data = msgpack.load(f)

            self.imgs_to_label = {Path(filename).stem: label for filename, label, *_ in data}
            self.imgs_to_author = {Path(filename).stem: self.author_fn(filename, *d) for filename, *d in data}

            self.imgs = [Path(path, 'lines') / filename for filename, *_ in data]
            assert len(self.imgs) > 0, f'No images found in {path}'

            img_sizes_path = Path(path, f'{nameset}_img_sizes.msgpack')
            self.imgs_to_sizes = self.load_img_sizes(img_sizes_path)
            assert set(self.imgs_to_label.keys()) == set(self.imgs_to_sizes.keys())

        if pkl_path is not None and not pkl_path.exists():
            self.save_pickle(pkl_path)

        charset_path = Path(path, 'charset.msgpack')
        with open(charset_path, 'rb') as f:
            charset = msgpack.load(f, strict_map_key=False)
        self.char_to_idx = charset['char2idx']
        self.idx_to_char = charset['idx2char']

        if max_width and max_height:
            target_width = {filename: width * max_height / height for filename, (width, height) in self.imgs_to_sizes.items()}
            if self.preloaded:
                self.imgs_preloaded = [img for path, img in zip(self.imgs, self.imgs_preloaded) if target_width[path.stem] <= max_width]
            self.imgs = [img for img in self.imgs if target_width[img.stem] <= max_width]

        self.imgs_set = set(self.imgs)
        self.imgs_to_idx = {img: idx for idx, img in enumerate(self.imgs)}
        authors = set(self.imgs_to_author.values())
        self.author_to_imgs = {author: {img for img in self.imgs if self.imgs_to_author[img.stem] == author} for author in authors}


class Norhand_dataset(Msgpack_dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, author_code='norhand', **kwargs)


class Rimes_dataset(Msgpack_dataset):
    def __init__(self, *args, **kwargs):
        def author_fn(filename, *args):
            return filename.split('-')[1]
        super().__init__(*args, author_code='rimes', author_fn=author_fn, **kwargs)


class ICFHR16_dataset(Msgpack_dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, author_code='icfh16', **kwargs)


class ICFHR14_dataset(Msgpack_dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, author_code='icfh14', **kwargs)


class LAM_dataset(Msgpack_dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, author_code='lam', **kwargs)


class Rodrigo_dataset(Msgpack_dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, author_code='rodrigo', **kwargs)


class SaintGall_dataset(Msgpack_dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, author_code='saintgall', **kwargs)


class Washington_dataset(Msgpack_dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, author_code='washington', **kwargs)


class Leopardi_dataset(Msgpack_dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, author_code='leopardi', **kwargs)


class MergedDataset(Dataset):
    def __init__(self, datasets, idx_to_char=None):
        super().__init__()
        self.datasets = datasets
        if idx_to_char:
            self.idx_to_char = idx_to_char
            self.char_to_idx = dict(zip(self.idx_to_char.values(), self.idx_to_char.keys()))
        else:
            self.char_to_idx = get_alphabet([''.join(list(d.idx_to_char.values())) for d in datasets])
            self.idx_to_char = dict(zip(self.char_to_idx.values(), self.char_to_idx.keys()))
        for dataset in self.datasets:
            dataset.char_to_idx = self.char_to_idx
            dataset.idx_to_char = self.idx_to_char

        authors = sorted(set(self.authors))
        self.local_author_to_global = {author: i for i, author in enumerate(authors)}
        


    @property
    def labels(self):
        return [label for dataset in self.datasets for label in dataset.imgs_to_label.values()]

    @property
    def vocab_size(self):
        return len(self.char_to_idx)

    @property
    def alphabet(self):
        return ''.join(sorted(self.char_to_idx.keys()))

    @property
    def imgs(self):
        return [img for dataset in self.datasets for img in dataset.imgs]
    
    @property
    def authors(self):
        return [f'db{i:02d}_{author}' for i, dataset in enumerate(self.datasets) for author in dataset.imgs_to_author.values()]

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx):
        for db_idx, dataset in enumerate(self.datasets):
            if idx < len(dataset):
                return self.add_author_idx(dataset[idx], db_idx)
            else:
                idx -= len(dataset)
        raise IndexError('Index out of range')
    

    def add_author_idx(self, sample, db_idx):
        tmp_dict = {}
        for key, val in sample.items():
            if key.endswith('author'):
                tmp_dict[key + '_idx'] = self.local_author_to_global[f'db{db_idx:02d}_{val}']
        sample.update(tmp_dict)
        return sample
    
    def batch_keys(self, *keys):
        if len(keys) == 1 and isinstance(keys[0], (list, tuple)):
            keys = keys[0]
        for dataset in self.datasets:
            dataset.batch_keys = keys

    def collate_fn(self, batch):
        collate_batch = {}

        for key in batch[0].keys():
            val = batch[0][key]
            if isinstance(val, torch.Tensor):
                collate_batch[key] = pad_images([sample[key] for sample in batch])
            elif isinstance(val, int):
                collate_batch[key] = torch.IntTensor([sample[key] for sample in batch])
            elif isinstance(val, float):
                collate_batch[key] = torch.FloatTensor([sample[key] for sample in batch])
            elif isinstance(val, bool):
                collate_batch[key] = torch.BoolTensor([sample[key] for sample in batch])
            else:
                collate_batch[key] = [sample[key] for sample in batch]

        return collate_batch


def dataset_factory(nameset, datasets, idx_to_char=None, img_height=64, gen_patch_width=8, gen_max_width=None,
                    img_channels=3, db_preload=False, pre_transform=None, post_transform=None, pkl_root='files/pkl_db', **kwargs):
    
    pre_transform = T.Compose([
        T.Convert(img_channels),
        T.ResizeFixedHeight(img_height),
        # T.FixedCharWidth(gen_patch_width),
        # T.ToTensor(),
        # T.PadNextDivisible(gen_patch_width),  # pad to next divisible of 16 (skip if already divisible)
    ]) if pre_transform is None else pre_transform

    post_transform = T.Compose([
        # T.ToPILImage(),
        # T.PadMinWidth(max(kwargs['style_patch_width'], kwargs['dis_patch_width']), padding_value=255),
        # T.RandomShrink(1.0, 1.0, min_width=max(kwargs['style_patch_width'], kwargs['dis_patch_width']), max_width=gen_max_width, snap_to=gen_patch_width),
        T.ToTensor(),
        # T.MedianRemove(),
        T.Normalize((0.5,), (0.5,))
    ]) if post_transform is None else post_transform

    datasets_list = []
    glob_kwargs = {'max_width': gen_max_width, 'max_height': img_height, 'transform': (pre_transform, post_transform), 'nameset': nameset, 'preload': db_preload}
    root_path = Path(kwargs['root_path'])
    pkl_root = Path(pkl_root)
    pkl_root.mkdir(parents=True, exist_ok=True)

    for name in tqdm(datasets, desc=f'Loading datasets {nameset}'):
        kwargs = {'pkl_path': pkl_root / f'{name.lower()}_{nameset}.pkl'}
        kwargs.update(glob_kwargs)
        if name.lower() == 'iam_words':
            datasets_list.append(IAM_dataset(root_path / 'IAM', dataset_type='words', **kwargs))
        elif name.lower() == 'iam_words_w3':
            datasets_list.append(IAM_dataset(root_path / 'IAM', dataset_type='words', min_text_width=3, **kwargs))
        elif name.lower() == 'iam_eval':
            datasets_list.append(IAM_eval(root_path / 'IAM', dataset_type='lines', **kwargs))
        elif name.lower() == 'iam_eval_sm':
            datasets_list.append(IAM_custom_eval(root_path / 'IAM', dataset_type='lines_sm', **kwargs))
        elif name.lower() == 'iam_eval_words':
            datasets_list.append(IAM_eval(root_path / 'IAM', dataset_type='words', **kwargs))
        elif name.lower() == 'iam_lines':
            datasets_list.append(IAM_dataset(root_path / 'IAM', dataset_type='lines', **kwargs))
        elif name.lower() == 'iam_lines_double':
            datasets_list.append(IAM_dataset_double(root_path / 'IAM', dataset_type='lines', **kwargs))
        elif name.lower() == 'iam_lines_16':
            datasets_list.append(IAM_dataset(root_path / 'IAM', dataset_type='lines_16', **kwargs))
        elif name.lower() == 'iam_lines_xl':
            datasets_list.append(IAM_custom_dataset(root_path / 'IAM', dataset_type='lines_xl', **kwargs))
        elif name.lower() == 'iam_lines_l':
            datasets_list.append(IAM_custom_dataset(root_path / 'IAM', dataset_type='lines_l', **kwargs))
        elif name.lower() == 'iam_lines_m':
            datasets_list.append(IAM_custom_dataset(root_path / 'IAM', dataset_type='lines_m', **kwargs))
        elif name.lower() == 'iam_lines_sm':
            datasets_list.append(IAM_custom_dataset(root_path / 'IAM', dataset_type='lines_sm', **kwargs))
        elif name.lower() == 'iam_lines_xs':
            datasets_list.append(IAM_custom_dataset(root_path / 'IAM', dataset_type='lines_xs', **kwargs))
        elif name.lower() == 'iam_lines_xxs':
            datasets_list.append(IAM_custom_dataset(root_path / 'IAM', dataset_type='lines_xxs', **kwargs))
        elif name.lower() == 'rimes':
            datasets_list.append(Rimes_dataset(root_path / 'Rimes', **kwargs))
        elif name.lower() == 'icfhr16':
            datasets_list.append(ICFHR16_dataset(root_path / 'ICFHR16', **kwargs))
        elif name.lower() == 'icfhr14':
            datasets_list.append(ICFHR14_dataset(root_path / 'ICFHR14', **kwargs))
        elif name.lower() == 'lam':
            datasets_list.append(LAM_dataset(root_path / 'LAM_msgpack', **kwargs))
        elif name.lower() == 'rodrigo':
            datasets_list.append(Rodrigo_dataset(root_path / 'Rodrigo', **kwargs))
        elif name.lower() == 'saintgall':
            datasets_list.append(SaintGall_dataset(root_path / 'SaintGall', **kwargs))
        elif name.lower() == 'washington':
            datasets_list.append(Washington_dataset(root_path / 'Washington', **kwargs))
        elif name.lower() == 'leopardi':
            datasets_list.append(Leopardi_dataset(root_path / 'LEOPARDI' / 'leopardi', **kwargs))
        elif name.lower() == 'norhand':
            datasets_list.append(Norhand_dataset(root_path / 'Norhand', **kwargs))
        else:
            raise ValueError(f'Unknown dataset {name}')
    return MergedDataset(datasets_list, idx_to_char)


if __name__ == '__main__':
    for nameset in ('train', 'val'):
        print(len(IAM_dataset('/mnt/ssd/datasets/IAM', nameset=nameset)[0]))
        print(len(SaintGall_dataset('/mnt/ssd/datasets/SaintGall', nameset=nameset)[0]))
        print(len(Norhand_dataset('/mnt/ssd/datasets/Norhand', nameset=nameset)[0]))
        print(len(Rimes_dataset('/mnt/ssd/datasets/Rimes', nameset=nameset)[0]))
        print(len(ICFHR16_dataset('/mnt/ssd/datasets/ICFHR16', nameset=nameset)[0]))
        print(len(ICFHR14_dataset('/mnt/ssd/datasets/ICFHR14', nameset=nameset)[0]))
        print(len(LAM_dataset('/mnt/ssd/datasets/LAM_msgpack', nameset=nameset)[0]))
        print(len(Rodrigo_dataset('/mnt/ssd/datasets/Rodrigo', nameset=nameset)[0]))
        print(len(Washington_dataset('/mnt/ssd/datasets/Washington', nameset=nameset)[0]))
        print(len(Leopardi_dataset('/mnt/ssd/datasets/LEOPARDI/leopardi', nameset=nameset)[0]))
