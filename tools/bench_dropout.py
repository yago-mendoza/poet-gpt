"""Why the real run was 3x slower than the benchmark: time one step with and without dropout.
Run it with the ORIGINAL nanoGPT model.py (attention dropout on) to reproduce the slowdown;
with this repo's model.py the attention-dropout setting has no effect.
Usage: python tools/bench_dropout.py
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import GPT, GPTConfig


def bench(dropout, attn_dropout_off=False):
    torch.manual_seed(0)
    m = GPT(GPTConfig(n_layer=6, n_head=8, n_embd=256, block_size=256, vocab_size=122, dropout=dropout, bias=False))
    if attn_dropout_off:
        for b in m.transformer.h:
            b.attn.dropout = 0.0
    m.train(); opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randint(0, 122, (32, 256)); y = torch.randint(0, 122, (32, 256))
    for i in range(8):
        if i == 3:
            t0 = time.time()
        _, loss = m(x, y); opt.zero_grad(); loss.backward(); opt.step()
    return (time.time() - t0) / 5 * 1000


print(f"dropout 0.0:                          {bench(0.0):.0f} ms/step")
print(f"dropout 0.2 everywhere:               {bench(0.2):.0f} ms/step")
print(f"dropout 0.2, none inside attention:   {bench(0.2, True):.0f} ms/step")
