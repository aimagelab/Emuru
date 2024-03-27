import string
import torch

from .constants import (
    START_OF_SEQUENCE,
    END_OF_SEQUENCE,
    PAD,
)


class Alphabet:
    def __init__(self, charset):
        self.extra = [PAD, START_OF_SEQUENCE, END_OF_SEQUENCE]

        charset_types = [self.extra, charset]

        self.char2idx = {}
        self.idx2char = {}
        self.labels = []
        current_id = 0
        for charset_type in charset_types:
            for char in charset_type:
                self.char2idx[char] = current_id
                self.idx2char[current_id] = char
                current_id += 1
                self.labels.append(char)

    def encode(self, x_in):
        out = []
        for i in x_in:
            out.append(self.char2idx[i])
        return torch.LongTensor(out)

    def _decode(self, x_in):
        out = []
        for i in x_in:
            out.append(self.idx2char[int(i)])
        return "".join(out)

    def decode(self, x_in, stopping_logits: list):
        """Decode a batch of logits into a list of strings."""
        text = []
        for b in x_in:
            stops = []
            for s in stopping_logits:
                stop = torch.where(b == s)[0]
                if len(stop) == 0:
                    stop = torch.LongTensor([len(b)])
                stops.append(stop[0])
            end_idx = torch.min(torch.stack(stops))
            text.append(self._decode(b[:end_idx]))
        return text
