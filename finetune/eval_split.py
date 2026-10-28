# Bits per character on the two parts of val: the paragraphs that share a verbatim run of 50+ characters with train
# (reposts) and the rest. python eval_split.py char|a0|a|b0|b
import json
import sys
from pathlib import Path

import numpy as np
import torch

from common import ROOT, bits_per_char

torch.set_num_threads(16)
which = sys.argv[1]
parts = {k: (ROOT / 'data_txt' / f'val_{k}.txt').read_text(encoding='utf-8') for k in ('clean', 'dirty')}
res = {}

if which == 'char':
    import pickle
    sys.path.insert(0, str(ROOT))
    from model import GPT, GPTConfig
    ck = torch.load(ROOT / 'out-poet-char/ckpt.pt', map_location='cpu', weights_only=False)
    m = GPT(GPTConfig(**ck['model_args']))
    m.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in ck['model'].items()})
    m.eval()
    stoi = pickle.load(open(ROOT / 'data/poet_char/meta.pkl', 'rb'))['stoi']
    for k, t in parts.items():
        ids = torch.tensor([stoi[c] for c in t], dtype=torch.long)
        res[k] = bits_per_char(lambda x: m(x, torch.zeros_like(x))[0], ids, len(t), 256, batch=32)
else:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = {'a0': 'DeepESP/gpt2-spanish', 'a': str(ROOT / 'out-option-a' / 'best'),
            'b0': 'Qwen/Qwen2.5-1.5B', 'b': 'Qwen/Qwen2.5-1.5B'}[which]
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    if which == 'b':
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, str(ROOT / 'out-option-b' / 'best')).merge_and_unload()
    m.eval()

    def lf(x):
        with torch.autocast('cpu', dtype=torch.bfloat16):
            return m(x).logits

    for k, t in parts.items():
        ids = torch.tensor(tok(t)['input_ids'], dtype=torch.long)
        res[k] = bits_per_char(lf, ids, len(t), 256, batch=8)
print(which, {k: round(v, 4) for k, v in res.items()}, flush=True)
out = ROOT / 'finetune' / 'eval_split.json'
d = json.loads(out.read_text()) if out.exists() else {}
d[which] = res
out.write_text(json.dumps(d, indent=1))
