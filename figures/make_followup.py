"""SVG figures for the follow-up piece, in the same style as make_figures.py (light and dark)."""
import json
import math
import statistics
from pathlib import Path

from make_figures import OUT, legend, svg, text

ROOT = Path(__file__).parent.parent


def read_log(path):
    pts = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        d = json.loads(line)
        if 'val_bpc' in d:
            pts.append((d['step'], d['val_bpc']))
    return sorted(set(pts))


# 1. Bits per character against passes over the training text -------------------------------------------------
def bpc_curves():
    a = read_log(ROOT / 'out-option-a' / 'log.jsonl')
    b = read_log(ROOT / 'out-option-b' / 'log.jsonl')
    a_steps, b_steps = 778, 437
    W, H = 760, 456
    x0, x1, y0, y1 = 70, 600, 360, 80          # plot box
    lo, hi = 0.90, 1.80
    X = lambda p: x0 + (x1 - x0) * p / 2.0
    Y = lambda v: y0 - (y0 - y1) * (v - lo) / (hi - lo)
    body = [text(24, 34, 'Bits per character on the 197 held-out posts, as each model trains', 't1', 17, weight=600),
            text(24, 56, 'Lower is better. Both pretrained models start below where the character model ended.', 't2', 12.5)]
    for v in (1.0, 1.2, 1.4, 1.6, 1.8):  # noqa
        body.append(f'<line class="grid" x1="{x0}" x2="{x1}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke-width="1"/>')
        body.append(text(x0 - 8, Y(v) + 4, f'{v:.1f}', 't3', 11, 'end'))
    for p in (0, 0.5, 1, 1.5, 2):
        body.append(text(X(p), y0 + 18, f'{p:g}', 't3', 11, 'middle'))
    body.append(text((x0 + x1) / 2, y0 + 38, 'passes over the training text', 't2', 12, 'middle'))
    body.append(f'<line class="axis" x1="{x0}" x2="{x1}" y1="{y0}" y2="{y0}" stroke-width="1"/>')
    # the character model's end point, after 7.6 passes, as a reference line
    body.append(f'<line class="axis" x1="{x0}" x2="{x1}" y1="{Y(1.7201):.1f}" y2="{Y(1.7201):.1f}" stroke-width="1.5" stroke-dasharray="5 4"/>')
    body.append(text(x1 + 8, Y(1.7201) + 4, 'character model', 't2', 11.5))
    body.append(text(x1 + 8, Y(1.7201) + 18, '1.72, after 7.6 passes', 't3', 11))
    # predicted bands at the end of training
    for lo_b, hi_b, cls, dx in ((1.18, 1.33, 's1', -14), (0.92, 1.08, 's2', -4)):
        yt, yb = Y(min(hi_b, hi)), Y(max(lo_b, lo))
        body.append(f'<rect class="{cls}" x="{X(2) + dx:.1f}" y="{yt:.1f}" width="8" height="{yb - yt:.1f}" rx="2" fill-opacity=".22"/>')
    for pts, steps, cls, name in ((a, a_steps, 's1', 'Option A'), (b, b_steps, 's2', 'Option B')):
        d = ' '.join(f'{"M" if i == 0 else "L"}{X(2 * s / steps):.1f},{Y(v):.1f}' for i, (s, v) in enumerate(pts))
        body.append(f'<path class="{cls}s" d="{d}" fill="none" stroke-width="2"/>')
        for s, v in (pts[0], pts[-1]):
            body.append(f'<circle class="{cls}" cx="{X(2 * s / steps):.1f}" cy="{Y(v):.1f}" r="4.5" stroke="var(--bg,#fff)" stroke-width="0"/>')
        dy = -8 if cls == 's1' else 46
        body.append(text(X(0) + 10, Y(pts[0][1]) + dy, f'{pts[0][1]:.2f} before training', 't2', 11.5))
        body.append(text(x1 + 8, Y(pts[-1][1]) + 4, f'{name}: {pts[-1][1]:.2f}', 't1', 12, weight=600))
    body.append(text(x1 + 8, Y(1.33) - 26, 'bars: the predicted', 't3', 11))
    body.append(text(x1 + 8, Y(1.33) - 12, 'ranges, written first', 't3', 11))
    body += legend(x0, H - 16, [('Option A: GPT2-Spanish, every weight, laptop CPU', 's1'), ('Option B: Qwen2.5-1.5B, LoRA, RTX 4090', 's2')])
    svg('bpc-curves.svg', W, H, 'Bits per character against passes over the training text for both options', body)


# 2. Longest verbatim copy per sample -------------------------------------------------------------------------
def verbatim():
    m = json.loads((ROOT / 'finetune' / 'memo.json').read_text(encoding='utf-8'))
    order = [('poet (held-out posts)', 'the poet, unseen posts', 'mute'),
             ('char model', 'character model', 'mute'),
             ('Option A (GPT2-Spanish)', 'Option A', 's1'),
             ('Option B (LoRA)', 'Option B', 's2')]
    order = [o for o in order if o[0] in m]
    W, rowh = 760, 64
    x0, x1 = 190, 690
    top = 92
    H = top + rowh * len(order) + 70
    lo, hi = math.log10(8), math.log10(1000)
    X = lambda v: x0 + (x1 - x0) * (math.log10(max(v, 8)) - lo) / (hi - lo)
    body = [text(24, 34, 'The longest run each sample copies verbatim from the training text', 't1', 17, weight=600),
            text(24, 56, 'One dot per sample of 700 characters, 60 per row; the bar is the median. Log scale.', 't2', 12.5)]
    for v in (10, 20, 50, 100, 200, 500, 1000):
        body.append(f'<line class="grid" x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{top - 10}" y2="{top + rowh * len(order) - 10}" stroke-width="1"/>')
        body.append(text(X(v), top + rowh * len(order) + 8, f'{v}', 't3', 11, 'middle'))
    body.append(text((x0 + x1) / 2, top + rowh * len(order) + 28, 'characters copied in one run', 't2', 12, 'middle'))
    body.append(f'<line class="axis" x1="{X(50):.1f}" x2="{X(50):.1f}" y1="{top - 16}" y2="{top + rowh * len(order) - 6}" stroke-width="1.5" stroke-dasharray="4 3"/>')
    body.append(text(X(50) + 5, top - 18, '50: counted as a copy', 't3', 11))
    for i, (key, label, cls) in enumerate(order):
        r = m[key]
        yc = top + rowh * i + rowh / 2 - 10
        body.append(text(x0 - 14, yc - 2, label, 't1', 12.5, 'end', weight=600))
        body.append(text(x0 - 14, yc + 14, f'{r["samples_with_copy"]} of {r["n"]} copy 50+', 't3', 11, 'end'))
        vals = sorted(r['longest_each'])
        seen = {}
        for v in vals:
            k = round(X(v))
            n = seen.get(k, 0)
            seen[k] = n + 1
            dy = (n % 5 - 2) * 5
            body.append(f'<circle class="{cls}" cx="{X(v):.1f}" cy="{yc + dy:.1f}" r="3.4" fill-opacity=".75"/>')
        med = statistics.median(vals)
        body.append(f'<rect class="t1" x="{X(med) - 1.5:.1f}" y="{yc - 16:.1f}" width="3" height="32" rx="1"/>')
        body.append(text(X(med), yc - 20, f'{med:g}', 't1', 11, 'middle'))
    svg('verbatim.svg', W, H, 'Longest verbatim copy from the training text per sample, by model', body)


if __name__ == '__main__':
    import sys
    for name in sys.argv[1:] or ['bpc_curves', 'verbatim']:
        globals()[name]()
