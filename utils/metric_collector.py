import torch
import numpy as np


class MetricCollector:
    def __init__(self):
        self.data = {}

    def __getitem__(self, item):
        avg = self.data[item]['tot'] / self.data[item]['count']
        return avg

    def __setitem__(self, key, value):
        if isinstance(key, (list, tuple)):
            assert isinstance(value, (list, tuple))
            assert len(key) == len(value)
            for k, v in zip(key, value):
                self._single_setitem(k, v)
        else:
            self._single_setitem(key, value)

    def _single_setitem(self, key, value):
        if isinstance(value, torch.Tensor):
            value = value.item()
        if np.isnan(value) or np.isinf(value):
            return
        if key not in self.data:
            self.data[key] = {'tot': 0.0, 'count': 0}
        self.data[key]['tot'] += value
        self.data[key]['count'] += 1

    def __contains__(self, item):
        return item in self.data

    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        assert isinstance(other, MetricCollector)
        new = MetricCollector()
        new.data = self.data | other.data
        return new
    
    def update(self, other):
        assert isinstance(other, dict)
        for k, v in other.items():
            self._single_setitem(k, v)

    def reset(self):
        self.data = {}

    def print(self, prefix='', suffix=''):
        msg = ' | '.join([f'{k}: {self[k]:.4f}' for k in self.data.keys()])
        print(prefix + msg + suffix)

    def dict(self):
        return {k: self[k] for k in self.data.keys()}

    def pytorch_tensor(self):
        return torch.tensor([[self.data[k]['tot'], self.data[k]['count']] for k in sorted(self.data.keys())])

    def load_pytorch_tensor(self, tensor):
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape[0] == len(self.data) and tensor.shape[1] == 2
        self.data = {k: {'tot': tot, 'count': count} for k, (tot, count) in zip(sorted(self.data.keys()), tensor.tolist())}
        return self