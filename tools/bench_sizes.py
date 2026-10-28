"""Training speed of three model sizes on this CPU (random data: measures the arithmetic only).
Usage: python tools/bench_sizes.py
"""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import GPT, GPTConfig

torch.manual_seed(0)
sizes = [
    ("README CPU recipe", 4, 4, 128, 64, 12),
    ("this run (mid)", 6, 8, 256, 256, 32),
    ("Shakespeare baby GPT", 6, 6, 384, 256, 64),
]
for name, L, H, E, T, B in sizes:
    m = GPT(GPTConfig(n_layer=L, n_head=H, n_embd=E, block_size=T, vocab_size=122, dropout=0.0, bias=False))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    x = torch.randint(0, 122, (B, T)); y = torch.randint(0, 122, (B, T))
    for i in range(13):
        if i == 3:
            t0 = time.time()
        _, loss = m(x, y); opt.zero_grad(); loss.backward(); opt.step()
    dt = (time.time() - t0) / 10
    print(f"{name:22} {m.get_num_params()/1e6:5.2f}M params  {dt*1000:6.0f} ms/step  {B*T/dt:8,.0f} tokens/s")
