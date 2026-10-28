"""Measure a corpus before training: size, character vocabulary and GPT-2 token count.
Usage: python tools/analyze_corpus.py <folder with .txt files>
"""
import sys
import unicodedata
from collections import Counter
from pathlib import Path

import tiktoken

src = Path(sys.argv[1])
files = sorted(src.rglob("*.txt"))
texts = [f.read_text(encoding="utf-8", errors="replace").strip() for f in files]
texts = [t for t in texts if t]
text = "\n\n".join(texts)

print(f"files: {len(files)}  (non-empty: {len(texts)})")
print(f"characters: {len(text):,}")
print(f"words: {len(text.split()):,}")
lens = sorted(len(t) for t in texts)
print(f"length per text: median {lens[len(lens)//2]}, max {lens[-1]}")

# as in the lecture: the distinct characters are the vocabulary
chars = sorted(set(text))
print(f"\nvocabulary (distinct characters): {len(chars)}")
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)
print("encode('hola') ->", encode("hola"))
print("decode(...)    ->", decode(encode("hola")))

freq = Counter(text)
rare = [c for c in chars if freq[c] < 5]
print(f"characters seen fewer than 5 times: {len(rare)}")
print("the 40 rarest:", " ".join(sorted(rare, key=lambda c: freq[c])[:40]))
print("the 100 most frequent:", repr("".join(c for c, _ in freq.most_common(100))))
symbols = sum(n for c, n in freq.items() if unicodedata.category(c) == "So")
print(f"symbol/emoji characters: {symbols:,}")
print(f"@ mentions: {text.count('@'):,}  # hashtags: {text.count('#'):,}")

enc = tiktoken.get_encoding("gpt2")
n_bpe = len(enc.encode_ordinary(text))
print(f"\nGPT-2 BPE tokens: {n_bpe:,}  ({len(text)/n_bpe:.2f} characters per token)")
