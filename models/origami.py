import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import warnings
import numpy as np
from einops import rearrange

from torch.utils.checkpoint import checkpoint


class CTCLabelConverter(nn.Module):
    def __init__(self, charset):
        super(CTCLabelConverter, self).__init__()
        self.charset = sorted(set(charset))

        # NOTE: 0 is reserved for 'blank' token required by CTCLoss
        self.dict = {char: i + 1 for i, char in enumerate(self.charset)}
        self.charset.insert(0, '[blank]')  # dummy '[blank]' token for CTCLoss (index 0)

    def encode(self, labels, device='cpu'):
        assert set(''.join(labels)) <= set(self.charset), f'The following character are not in charset {set("".join(labels)) - set(self.charset)}'
        length = torch.LongTensor([len(lbl) for lbl in labels])
        labels = [torch.LongTensor([self.dict[char] for char in lbl]) for lbl in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
        return labels.to(device), length.to(device)

    def decode(self, labels, length):
        texts = []
        assert len(labels) == len(length)
        for lbl, lbl_len in zip(labels, length):
            char_list = []
            for i in range(lbl_len):
                if lbl[i] != 0 and (not (i > 0 and lbl[i - 1] == lbl[i])) and lbl[i] < len(self.charset):  # removing repeated characters and blank.
                    char_list.append(self.charset[lbl[i]])
            texts.append(''.join(char_list))
        return texts

    def decode_batch(self, preds):
        preds = rearrange(preds, 'w b c -> b w c')
        _, preds_index = preds.max(2)
        preds_index = preds_index.cpu().numpy()
        preds_size = preds.size(1) - (np.flip(preds_index, 1) > 0).argmax(-1)
        preds_size = np.where(preds_size < preds.size(1), preds_size, 0)
        return self.decode(preds_index, preds_size)


def checkpoint_sequential_step(functions, segments, *inputs, **kwargs):
    r"""A helper function for checkpointing sequential models.

    Sequential models execute a list of modules/functions in order
    (sequentially). Therefore, we can divide such a model in various segments
    and checkpoint each segment. All segments except the last will run in
    :func:`torch.no_grad` manner, i.e., not storing the intermediate
    activations. The inputs of each checkpointed segment will be saved for
    re-running the segment in the backward pass.

    See :func:`~torch.utils.checkpoint.checkpoint` on how checkpointing works.

    .. warning::
        Checkpointing doesn't work with :func:`torch.autograd.grad`, but only
        with :func:`torch.autograd.backward`.

    .. warning:
        At least one of the inputs needs to have :code:`requires_grad=True` if
        grads are needed for model inputs, otherwise the checkpointed part of the
        model won't have gradients.

    Args:
        functions: A :class:`torch.nn.Sequential` or the list of modules or
            functions (comprising the model) to run sequentially.
        segments: Number of chunks to create in the model
        inputs: tuple of Tensors that are inputs to :attr:`functions`
        preserve_rng_state(bool, optional, default=True):  Omit stashing and restoring
            the RNG state during each checkpoint.

    Returns:
        Output of running :attr:`functions` sequentially on :attr:`*inputs`

    Example:
        >>> model = nn.Sequential(...)
        >>> input_var = checkpoint_sequential(model, chunks, input_var)
    """
    # Hack to mix *args with **kwargs in a python 2.7-compliant way
    preserve = kwargs.pop('preserve_rng_state', True)
    if kwargs:
        raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))

    # To accept variadic arguments is not consistent with nn.Sequential.
    # This interface will be changed at PyTorch 1.3.
    # See also: https://github.com/pytorch/pytorch/issues/19260
    if not inputs:
        warnings.warn('Giving no input to checkpoint_sequential has been deprecated, '
                      'a TypeError will be raised after PyTorch 1.3',
                      DeprecationWarning)
    elif len(inputs) > 1:
        warnings.warn('multiple inputs to checkpoint_sequential has been deprecated, '
                      'a TypeError will be raised after PyTorch 1.3',
                      DeprecationWarning)

    def run_function(start, end, functions):
        def forward(*inputs):
            for j in range(start, end + 1):
                if isinstance(inputs, tuple):
                    inputs = functions[j](*inputs)
                else:
                    inputs = functions[j](inputs)
            return inputs
        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = list(functions.children())

    segment_size = segments
    # the last chunk has to be non-volatile
    end = -1
    for start in range(0, len(functions)-segments, segments):
        end = start + segment_size - 1
        inputs = checkpoint(run_function(start, end, functions), *inputs,
                            preserve_rng_state=preserve)
        if not isinstance(inputs, tuple):
            inputs = (inputs,)

    return checkpoint(run_function(end + 1, len(functions) - 1, functions), *inputs)


class LN(nn.Module):
    def forward(self, x):
        return F.layer_norm(x, x.size()[1:], weight=None, bias=None, eps=1e-05)

# @gin.configurable


class PadPool(nn.Module):
    def forward(self, x):
        x = F.pad(x, [0, 0, 0, 1])
        x = F.max_pool2d(x, (2, 2), stride=(1, 2))
        return x


def pCnv(inp, out, groups=1):
    return nn.Sequential(
        nn.Conv2d(inp, out, 1, bias=False, groups=groups),
        nn.InstanceNorm2d(out, affine=True)
    )

# regarding same padding in PT https://github.com/pytorch/pytorch/issues/3867


def dsCnv(inp, k):
    return nn.Sequential(
        nn.Conv2d(inp, inp, k, groups=inp, bias=False, padding=(k - 1) // 2),
        nn.InstanceNorm2d(inp, affine=True)
    )


ngates = 2


class Gate(nn.Module):
    def __init__(self, ifsz):
        super().__init__()
        self.ln = LN()

    def forward(self, x):
        t0, t1 = torch.chunk(x, ngates, dim=1)
        t0 = torch.tanh(t0)
        t1.sub(2)
        t1 = torch.sigmoid(t1)
        return t1*t0


def customGC(module):
    def custom_forward(*inputs):
        inputs = module(inputs[0])
        return inputs
    return custom_forward

# @gin.configurable


class GateBlock(nn.Module):
    def __init__(self, ifsz, ofsz, gt=True, ksz=3, GradCheck=None):  # GradCheck=gin.REQUIRED
        super().__init__()

        cfsz = int(math.floor(ifsz/2))
        ifsz2 = ifsz + ifsz % 2

        self.sq = nn.Sequential(
            pCnv(ifsz, cfsz),
            dsCnv(cfsz, ksz),
            nn.ELU(),
            ###########
            pCnv(cfsz, cfsz*ngates),
            dsCnv(cfsz*ngates, ksz),
            Gate(cfsz),
            ###########
            pCnv(cfsz, ifsz),
            dsCnv(ifsz, ksz),
            nn.ELU(),
        )

        self.gt = gt
        self.gc = GradCheck

    def forward(self, x):
        if self.gc >= 1:
            y = checkpoint(customGC(self.sq), x)
        else:
            y = self.sq(x)

        out = x + y
        return out

# @gin.configurable


class InitBlock(nn.Module):
    def __init__(self, fup, n_channels):
        super().__init__()

        self.n1 = LN()
        self.Initsq = nn.Sequential(
            pCnv(n_channels, fup),
            nn.Softmax(dim=1),
            dsCnv(fup, 11),
            LN()
        )

    def forward(self, x):
        x = self.n1(x)
        xt = x
        x = self.Initsq(x)
        x = torch.cat([x, xt], 1)
        return x


class OrigamiNet(nn.Module):
    def __init__(self, o_classes, n_channels=3, wmul=1.0, lreszs=None, lszs=None, nlyrs=12, fup=33, GradCheck=0, reduceAxis=3, **kwargs):
        super().__init__()

        self.lreszs = lreszs
        self.Initsq = InitBlock(fup, n_channels)

        if lreszs is None:
            lreszs = {
                0: torch.nn.MaxPool2d((2, 2)),
                2: torch.nn.MaxPool2d((2, 2)),
                4: torch.nn.MaxPool2d((2, 2)),
                6: PadPool(),
                8: PadPool(),
                10: torch.nn.Upsample(size=(450, 15), mode='bilinear', align_corners=True),
                11: torch.nn.Upsample(size=(1100, 8), mode='bilinear', align_corners=True)
            }

        if lszs is None:
            lszs = {0: 128, 2: 256, 4: 512, 11: 256}

        layers = []
        isz = fup + n_channels
        osz = isz
        for i in range(nlyrs):
            osz = int(math.floor(lszs[i] * wmul)) if i in lszs else isz
            layers.append(GateBlock(isz, osz, True, 3, GradCheck=GradCheck))

            if isz != osz:
                layers.append(pCnv(isz, osz))
                layers.append(nn.ELU())
            isz = osz

            if i in lreszs:
                layers.append(lreszs[i])

        layers.append(LN())
        self.Gatesq = nn.Sequential(*layers)

        self.Finsq = nn.Sequential(
            pCnv(osz, o_classes),
            nn.ELU(),
        )

        self.n1 = LN()
        self.it = 0
        self.gc = GradCheck
        self.reduceAxis = reduceAxis

    def forward(self, x):
        x = self.Initsq(x)

        if self.gc >= 2:
            x = checkpoint_sequential_step(self.Gatesq, 4, x)  # slower, more memory save
            # x = checkpoint_sequential_step(self.Gatesq,8,x)  #faster, less memory save
        else:
            x = self.Gatesq(x)

        x = self.Finsq(x)

        x = torch.mean(x, self.reduceAxis, keepdim=False)
        x = self.n1(x)
        x = x.permute(0, 2, 1)

        return x
