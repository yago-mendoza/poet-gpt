# How much of the training text each model copies, verbatim.
# For every position of a sample, the longest run that starts there and occurs verbatim in train (matching
# statistics: the run at i+1 is at least the run at i minus one, so it is enough to extend from there).
# Control: pieces of 700 characters of the real held-out posts, which no model saw.
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import ROOT, texts

SEP = '\n---------------\n'
K = 50  # a copy: 50 characters in a row or more


def load(path):
    t = Path(path).read_text(encoding='utf-8')
    parts = [p.strip('\n') for p in t.split(SEP)]
    return [p for p in parts if len(p) >= 200]


def char_samples(path):
    # nanoGPT's sample.py prints headers before the first sample and separates samples with a line of dashes
    t = Path(path).read_text(encoding='utf-8')
    parts = t.split('\n---------------')
    return [p.strip('\n') for p in parts if len(p.strip()) >= 200 and 'number of parameters' not in p[:200]]


def matching(s, T):
    ms, L = [0] * len(s), 0
    for i in range(len(s)):
        L = max(L - 1, 0)
        while i + L < len(s) and s[i:i + L + 1] in T:
            L += 1
        ms[i] = L
    return ms


def stats(samples, T):
    rows = []
    for s in samples:
        ms = matching(s, T)
        covered = [False] * len(s)
        for i, m in enumerate(ms):
            if m >= K:
                for j in range(i, i + m):
                    covered[j] = True
        i = max(range(len(s)), key=lambda k: ms[k])
        rows.append({'chars': len(s), 'longest': ms[i], 'longest_text': s[i:i + ms[i]],
                     'copied_share': sum(covered) / len(s),
                     'copies': sum(1 for k, m in enumerate(ms) if m >= K and (k == 0 or ms[k - 1] < K))})
    L = sorted(r['longest'] for r in rows)
    total = sum(r['chars'] for r in rows)
    return {
        'n': len(rows), 'chars': total,
        'longest_median': statistics.median(L), 'longest_p90': L[int(0.9 * (len(L) - 1))], 'longest_max': L[-1],
        'samples_with_copy': sum(1 for r in rows if r['longest'] >= K),
        'copied_share': sum(r['copied_share'] * r['chars'] for r in rows) / total,
        'copies_per_10k_chars': 1e4 * sum(r['copies'] for r in rows) / total,
        'longest_each': [r['longest'] for r in rows],
        'top': sorted(rows, key=lambda r: -r['longest'])[:3],
    }


if __name__ == '__main__':
    train, val = texts()
    posts = [p for p in val.split('\n\n') if p.strip()]
    # control: 60 pieces of 700 characters of val, spread over the whole text
    step = len(val) // 60
    control = [val[i * step:i * step + 700] for i in range(60)]
    sets = {'poet (held-out posts)': control,
            'char model': char_samples(ROOT / 'samples_char60.txt'),
            'Qwen2.5-1.5B, no training': load(ROOT / 'out-option-b' / 'samples_base60.txt'),
            'Option B (LoRA)': load(ROOT / 'out-option-b' / 'samples_b60.txt')}
    a = ROOT / 'out-option-a' / 'samples_a60.txt'
    if a.exists():
        sets['Option A (GPT2-Spanish)'] = load(a)
    out = {}
    for name, ss in sets.items():
        out[name] = stats(ss, train)
        r = out[name]
        print(f"{name:28} n={r['n']:3}  longest median {r['longest_median']:5}  p90 {r['longest_p90']:5}  max {r['longest_max']:5}"
              f"  samples with a {K}+ copy {r['samples_with_copy']:2}  copied {100 * r['copied_share']:5.1f}%"
              f"  copies/10k chars {r['copies_per_10k_chars']:.1f}", flush=True)
    (ROOT / 'finetune' / 'memo.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
