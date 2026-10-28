# Removes from the training text every paragraph that repeats 50 or more characters in a row of an earlier one
# (reposts), so that none is seen several times per pass. val is left alone.
import hashlib
from pathlib import Path

from common import ROOT, texts

K = 50
train, _ = texts()
paras = train.split('\n\n')
seen, keep, dropped = set(), [], []
for p in paras:
    grams = {hashlib.blake2b(p[i:i + K].encode(), digest_size=8).digest() for i in range(max(0, len(p) - K + 1))}
    if grams & seen:
        dropped.append(p)
        continue
    seen |= grams
    keep.append(p)
out = '\n\n'.join(keep)
(ROOT / 'data_txt' / 'train_dedup.txt').write_text(out, encoding='utf-8')
dc = sum(len(p) for p in dropped)
print(f'paragraphs {len(paras):,}: kept {len(keep):,}, dropped {len(dropped):,}')
print(f'characters {len(train):,}: kept {len(out):,}, dropped {dc:,} ({100 * dc / len(train):.1f}%)')
