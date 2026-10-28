"""Does PyTorch's fused CPU attention kernel accept dropout? Time one attention call either way.
Usage: python tools/check_sdpa_kernel.py
"""
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

q = torch.randn(32, 8, 256, 32, requires_grad=True)
k = torch.randn_like(q, requires_grad=True)
v = torch.randn_like(q, requires_grad=True)
for p in (0.0, 0.2):
    try:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=True)
        print(f"dropout_p={p}: fused CPU kernel OK")
    except RuntimeError as e:
        print(f"dropout_p={p}: fused CPU kernel refused: {str(e).splitlines()[0][:100]}")
    t0 = time.time()
    for _ in range(5):
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=p, is_causal=True)
        y.sum().backward()
    print(f"   default dispatch, forward + backward: {(time.time() - t0) / 5 * 1000:.0f} ms")
