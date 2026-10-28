"""
Prepare a folder of .txt files (one post per file) for character-level nanoGPT.
Usage: python data/poet_char/prepare.py <folder with .txt files>
Writes train.bin, val.bin (uint16) and meta.pkl (vocab_size, stoi, itos) next to this script.
"""
import os
import pickle
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

src = Path(sys.argv[1])
out = os.path.dirname(__file__)

# 1. read the posts (one per file); the originals are never modified
files = sorted(src.rglob("*.txt"))
posts = [f.read_text(encoding="utf-8", errors="replace") for f in files]

# 2. clean: drop @mentions and #hashtags, normalise whitespace
mention = re.compile(r"@[\w.]*\w")
hashtag = re.compile(r"#\w+")
def clean(t):
    t = mention.sub("", t)
    t = hashtag.sub("", t)
    t = t.replace("\xa0", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r" +([.,;:!?)])", r"\1", t)
    t = "\n".join(line.strip() for line in t.splitlines())
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()
posts = [clean(t) for t in posts]
posts = [t for t in posts if t]

# 3. drop characters seen fewer than 5 times (rare emoji, stray scripts...)
freq = Counter("".join(posts))
rare = {c for c, n in freq.items() if n < 5}
posts = ["".join(c for c in t if c not in rare) for t in posts]
print(f"posts: {len(posts)}, rare characters removed: {len(rare)}")

# 4. 90/10 split by whole posts, shuffled with a fixed seed
random.Random(1337).shuffle(posts)
n_val = len(posts) // 10
val_text = "\n\n".join(posts[:n_val])
train_text = "\n\n".join(posts[n_val:])

# 5. vocabulary: the distinct characters, and the character <-> integer maps
chars = sorted(set(train_text + val_text))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)
print(f"vocab_size: {vocab_size}")
print(f"train: {len(train_text):,} characters  val: {len(val_text):,} characters")

train_ids = np.array(encode(train_text), dtype=np.uint16)
val_ids = np.array(encode(val_text), dtype=np.uint16)
train_ids.tofile(os.path.join(out, "train.bin"))
val_ids.tofile(os.path.join(out, "val.bin"))
with open(os.path.join(out, "meta.pkl"), "wb") as f:
    pickle.dump({"vocab_size": vocab_size, "itos": itos, "stoi": stoi}, f)

# 6. the lecture's example: with block_size = 8, one chunk holds 8 training examples
block_size = 8
x = train_ids[: block_size]
y = train_ids[1: block_size + 1]
print(f"\nexample with block_size={block_size}: {decode(x.tolist())!r}")
for t in range(block_size):
    print(f"  context {decode(x[:t+1].tolist())!r:14} -> next {decode([y[t]])!r}")
