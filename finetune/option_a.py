# Option A: a GPT-2 small trained from scratch on Spanish (DeepESP/gpt2-spanish, 124M), tuned whole on the CPU.
# python finetune/option_a.py bench   starting loss and seconds per step, measured before predicting anything
# python finetune/option_a.py train   the fine-tuning run, evaluated in bits per character every 100 steps
# On a GPU: --data <dir with train.txt and val.txt> [--train-file train_dedup.txt] [--out dir]
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import ROOT, bits_per_char

NAME = 'DeepESP/gpt2-spanish'
BLOCK, BATCH = 256, 8
LR, MIN_LR, WARMUP = 1e-4, 1e-5, 40
EPOCHS = 2
EVAL_EVERY = 100

ap = argparse.ArgumentParser()
ap.add_argument('mode')
ap.add_argument('--data')
ap.add_argument('--train-file', default='train.txt')
ap.add_argument('--out', default=str(ROOT / 'out-option-a'))
args = ap.parse_args()
mode, OUT = args.mode, Path(args.out)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(1337)
torch.set_num_threads(16)

tok = AutoTokenizer.from_pretrained(NAME)
# An explicit dtype: transformers 5 loads the checkpoint in its float16, and on this CPU float16 has no fast path
# (one pass of 1 x 32 tokens: 6,459 ms in float16, 134 in float32, 57 in bfloat16, measured).
# No dropout inside attention: the fast CPU kernel refuses it (the first half of the write-up).
model = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float32, attn_pdrop=0.0, attn_implementation='sdpa').to(dev)
if args.data:
    train_text = (Path(args.data) / args.train_file).read_text(encoding='utf-8')
    val_text = (Path(args.data) / 'val.txt').read_text(encoding='utf-8')
else:
    from common import texts
    train_text, val_text = texts()
train_ids = torch.tensor(tok(train_text)['input_ids'], dtype=torch.long)
val_ids = torch.tensor(tok(val_text)['input_ids'], dtype=torch.long)
print(f'train {len(train_ids):,} tokens ({len(train_text) / len(train_ids):.2f} chars/token), val {len(val_ids):,} tokens', flush=True)

def logits_fn(x):
    # mixed precision: float32 weights, bfloat16 arithmetic (4.5 s per step against 6.5)
    with torch.autocast(dev, dtype=torch.bfloat16):
        return model(x.to(dev)).logits
steps = EPOCHS * len(train_ids) // (BLOCK * BATCH)


def get_batch():
    ix = torch.randint(len(train_ids) - BLOCK - 1, (BATCH,))
    x = torch.stack([train_ids[i:i + BLOCK] for i in ix])
    y = torch.stack([train_ids[i + 1:i + BLOCK + 1] for i in ix])
    return x, y


def lr_at(s):
    if s < WARMUP:
        return LR * (s + 1) / WARMUP
    return MIN_LR + 0.5 * (1 + math.cos(math.pi * (s - WARMUP) / (steps - WARMUP))) * (LR - MIN_LR)


def evaluate():
    model.eval()
    t = time.time()
    v = bits_per_char(logits_fn, val_ids, len(val_text), BLOCK)
    model.train()
    return v, time.time() - t


opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.99), weight_decay=0.1)
model.train()

if mode == 'bench':
    v, te = evaluate()
    print(f'bpc at step 0: {v:.3f}  (eval {te:.0f} s)', flush=True)
    times = []
    for s in range(6):
        t = time.time()
        x, y = get_batch()
        loss = torch.nn.functional.cross_entropy(logits_fn(x).float().reshape(-1, model.config.vocab_size), y.to(dev).reshape(-1))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        times.append(time.time() - t)
        print(f'step {s}: {times[-1]:.2f} s  loss {loss.item():.3f}', flush=True)
    print(f'steps planned: {steps}, mean of steps 1-5: {sum(times[1:]) / 5:.2f} s', flush=True)
    sys.exit()

OUT.mkdir(exist_ok=True)
log = open(OUT / 'log.jsonl', 'a')
best = float('inf')
t0 = time.time()
for s in range(steps + 1):
    if s % EVAL_EVERY == 0 or s == steps:
        v, te = evaluate()
        print(f'eval step {s}: val {v:.4f} bpc ({te:.0f} s, {(time.time() - t0) / 60:.1f} min)', flush=True)
        log.write(json.dumps({'step': s, 'val_bpc': v, 'min': (time.time() - t0) / 60}) + '\n')
        log.flush()
        if v < best:
            best = v
            model.save_pretrained(OUT / 'best')
            tok.save_pretrained(OUT / 'best')
        if s == steps:
            break
    for g in opt.param_groups:
        g['lr'] = lr_at(s)
    t = time.time()
    x, y = get_batch()
    loss = torch.nn.functional.cross_entropy(logits_fn(x).float().reshape(-1, model.config.vocab_size), y.to(dev).reshape(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    if s % 10 == 0:
        print(f'step {s}/{steps}: loss {loss.item():.3f}  {time.time() - t:.2f} s', flush=True)
        log.write(json.dumps({'step': s, 'train_loss': loss.item(), 'sec': time.time() - t}) + '\n')
        log.flush()
print(f'done in {(time.time() - t0) / 60:.1f} min, best val {best:.4f} bpc', flush=True)
