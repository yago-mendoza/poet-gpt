# Bits per character on the two parts of val (new paragraphs, and paragraphs that repeat train), on a GPU or CPU.
# python eval_parts.py <model> [--adapter dir] --data dir   (dir with val_clean.txt and val_dirty.txt)
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import bits_per_char

ap = argparse.ArgumentParser()
ap.add_argument('model')
ap.add_argument('--adapter')
ap.add_argument('--data', default='.')
ap.add_argument('--name', default='model')
args = ap.parse_args()
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
tok = AutoTokenizer.from_pretrained(args.model)
m = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(dev)
if args.adapter:
    from peft import PeftModel
    m = PeftModel.from_pretrained(m, args.adapter).merge_and_unload()
m.eval()


def lf(x):
    with torch.autocast(dev, dtype=torch.bfloat16):
        return m(x.to(dev)).logits


res = {}
for k in ('clean', 'dirty'):
    t = (Path(args.data) / f'val_{k}.txt').read_text(encoding='utf-8')
    res[k] = bits_per_char(lf, torch.tensor(tok(t)['input_ids']), len(t), 256, batch=16)
print(json.dumps({args.name: res}), flush=True)
