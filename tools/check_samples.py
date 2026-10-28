"""Usage: python tools/check_samples.py samples.txt (samples separated by nanoGPT's dashed lines).
How much of the samples is copied verbatim from the training text? Longest shared run per sample."""
import pickle, sys
import numpy as np
d = "data/poet_char"
itos = pickle.load(open(f"{d}/meta.pkl", "rb"))["itos"]
train = "".join(itos[int(i)] for i in np.fromfile(f"{d}/train.bin", dtype=np.uint16))
samples = [s.strip() for s in open(sys.argv[1], encoding="utf-8").read().split("---------------") if s.strip()]
for k, s in enumerate(samples):
    best, lo = "", 0
    for i in range(len(s)):  # grow the longest substring starting at i found in train
        L = len(best) + 1
        while i + L <= len(s) and s[i:i + L] in train:
            best = s[i:i + L]; L += 1
    print(f"sample {k+1}: {len(s)} chars, longest verbatim run {len(best)}: {best!r}")
