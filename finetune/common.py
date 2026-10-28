# What the three runs share: the same train and val text as the character model, and one measure,
# bits per character over the whole validation text, in windows that do not overlap.
import math
import pickle
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data' / 'poet_char'


def texts():
    import numpy as np
    # decoding train.bin and val.bin guarantees exactly the same split of posts and the same cleaning
    meta = pickle.load(open(DATA / 'meta.pkl', 'rb'))
    itos = meta['itos']
    dec = lambda f: ''.join(itos[i] for i in np.fromfile(DATA / f, dtype=np.uint16).tolist())
    return dec('train.bin'), dec('val.bin')


@torch.no_grad()
def bits_per_char(logits_fn, ids, n_chars, block, batch=8):
    """ids: the tokenized val text as a 1D tensor. Every token but the first is predicted once, with as much context as
    fits in its window. Returns bits per character of the original text."""
    starts = list(range(0, len(ids) - 1, block))
    nll = 0.0
    for b in range(0, len(starts), batch):
        chunk = [ids[s:s + block + 1] for s in starts[b:b + batch]]
        L = max(len(c) for c in chunk)
        x = torch.zeros(len(chunk), L - 1, dtype=torch.long)
        y = torch.full((len(chunk), L - 1), -100, dtype=torch.long)
        for i, c in enumerate(chunk):
            x[i, :len(c) - 1] = c[:-1]
            y[i, :len(c) - 1] = c[1:]
        logits = logits_fn(x)
        nll += torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.to(logits.device).reshape(-1),
                                                 ignore_index=-100, reduction='sum').item()
    return nll / n_chars / math.log(2)
