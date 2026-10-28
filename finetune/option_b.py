# Option B: Qwen2.5-1.5B frozen, with LoRA adapters on its layers, on a rented GPU.
# python option_b.py bench|train [--model Qwen/Qwen2.5-1.5B] [--data dir with train.txt and val.txt] [--out dir]
import argparse
import json
import math
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument('mode')
p.add_argument('--model', default='Qwen/Qwen2.5-1.5B')
p.add_argument('--data', default='.')
p.add_argument('--out', default='out-option-b')
p.add_argument('--steps', type=int, default=0, help='0: two epochs')
p.add_argument('--rank', type=int, default=16)
p.add_argument('--train-file', default='train.txt')
args = p.parse_args()

BLOCK, BATCH, ACCUM = 256, 8, 2  # 16 sequences per step in two passes: logits over 152K tokens do not fit at once
LR, MIN_LR, WARMUP, EPOCHS, EVAL_EVERY = 2e-4, 2e-5, 30, 2, 50
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.bfloat16 if dev == 'cuda' else torch.float32
torch.manual_seed(1337)


@torch.no_grad()
def bits_per_char(model, ids, n_chars, block=BLOCK, batch=16):
    starts = list(range(0, len(ids) - 1, block))
    nll = 0.0
    for b in range(0, len(starts), batch):
        chunk = [ids[s:s + block + 1] for s in starts[b:b + batch]]
        L = max(len(c) for c in chunk)
        x = torch.zeros(len(chunk), L - 1, dtype=torch.long)
        y = torch.full((len(chunk), L - 1), -100, dtype=torch.long)
        for i, c in enumerate(chunk):
            x[i, :len(c) - 1] = c[:-1]
            y[i, :len(c) - 1] = c[1:]
        logits = model(x.to(dev)).logits.float()
        nll += torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), y.to(dev).reshape(-1),
                                                 ignore_index=-100, reduction='sum').item()
    return nll / n_chars / math.log(2)


tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(dev)
n_base = sum(p.numel() for p in model.parameters())
train_text = (Path(args.data) / args.train_file).read_text(encoding='utf-8')
val_text = (Path(args.data) / 'val.txt').read_text(encoding='utf-8')
train_ids = torch.tensor(tok(train_text)['input_ids'], dtype=torch.long)
val_ids = torch.tensor(tok(val_text)['input_ids'], dtype=torch.long)
print(f'{args.model}: {n_base / 1e6:.0f}M params; train {len(train_ids):,} tokens '
      f'({len(train_text) / len(train_ids):.2f} chars/token), val {len(val_ids):,}', flush=True)

model.eval()
t = time.time()
v0 = bits_per_char(model, val_ids, len(val_text))
print(f'bpc before any training: {v0:.3f} ({time.time() - t:.0f} s)', flush=True)

lora = LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type='CAUSAL_LM',
                  target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
model = get_peft_model(model, lora)
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'LoRA rank {args.rank}: {n_train:,} trainable ({100 * n_train / n_base:.2f}% of the base)', flush=True)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, betas=(0.9, 0.99), weight_decay=0.0)
steps = args.steps or EPOCHS * len(train_ids) // (BLOCK * BATCH * ACCUM)


def lr_at(s):
    if s < WARMUP:
        return LR * (s + 1) / WARMUP
    return MIN_LR + 0.5 * (1 + math.cos(math.pi * (s - WARMUP) / max(1, steps - WARMUP))) * (LR - MIN_LR)


def step(s):
    for g in opt.param_groups:
        g['lr'] = lr_at(s)
    total = 0.0
    for _ in range(ACCUM):
        ix = torch.randint(len(train_ids) - BLOCK - 1, (BATCH,))
        x = torch.stack([train_ids[i:i + BLOCK] for i in ix]).to(dev)
        y = torch.stack([train_ids[i + 1:i + BLOCK + 1] for i in ix]).to(dev)
        with torch.autocast(dev, dtype=dtype, enabled=dev == 'cuda'):
            logits = model(x).logits
        loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1)) / ACCUM
        loss.backward()
        total += loss.item()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)
    return total


model.train()
if args.mode == 'bench':
    for s in range(4):
        t = time.time()
        l = step(s)
        print(f'step {s}: {time.time() - t:.2f} s, loss {l:.3f}', flush=True)
    print(f'steps planned: {steps}', flush=True)
    raise SystemExit

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
log = open(out / 'log.jsonl', 'a')
log.write(json.dumps({'step': 0, 'val_bpc': v0, 'min': 0}) + '\n')
best, t0 = v0, time.time()
for s in range(steps):
    l = step(s)
    if s % 10 == 0:
        log.write(json.dumps({'step': s, 'train_loss': l}) + '\n')
    if (s + 1) % EVAL_EVERY == 0 or s + 1 == steps:
        model.eval()
        v = bits_per_char(model, val_ids, len(val_text))
        model.train()
        m = (time.time() - t0) / 60
        print(f'step {s + 1}/{steps}: train loss {l:.3f}, val {v:.4f} bpc, {m:.1f} min', flush=True)
        log.write(json.dumps({'step': s + 1, 'val_bpc': v, 'min': m}) + '\n')
        log.flush()
        if v < best:
            best = v
            model.save_pretrained(out / 'best')
print(f'done in {(time.time() - t0) / 60:.1f} min, best val {best:.4f} bpc', flush=True)
