"""Characters per token on the validation split: characters, GPT-2 BPE, and SentencePiece trained here.
Run after data/poet_char/prepare.py. Usage: python tools/measure_tokenizers.py
"""
import os
import pickle

import numpy as np
import sentencepiece as spm
import tiktoken

d = "data/poet_char"
meta = pickle.load(open(f"{d}/meta.pkl", "rb"))
itos = meta["itos"]
dec = lambda a: "".join(itos[int(i)] for i in a)
train = dec(np.fromfile(f"{d}/train.bin", dtype=np.uint16))
val = dec(np.fromfile(f"{d}/val.bin", dtype=np.uint16))
os.makedirs("tok", exist_ok=True)
open("tok/train.txt", "w", encoding="utf-8").write(train)

print(f"characters  vocab {len(itos):>6}  chars/token 1.00  val tokens {len(val):,}")
gpt2 = tiktoken.get_encoding("gpt2")
n = len(gpt2.encode_ordinary(val))
print(f"GPT-2 BPE   vocab {gpt2.n_vocab:>6}  chars/token {len(val)/n:.2f}  val tokens {n:,}")
for V in (1000, 4000, 8000):
    spm.SentencePieceTrainer.train(input="tok/train.txt", model_prefix=f"tok/sp{V}", vocab_size=V,
                                   model_type="bpe", character_coverage=1.0,
                                   max_sentence_length=100000, minloglevel=2)
    sp = spm.SentencePieceProcessor(model_file=f"tok/sp{V}.model")
    n = len(sp.encode(val))
    print(f"SP BPE      vocab {V:>6}  chars/token {len(val)/n:.2f}  val tokens {n:,}")
