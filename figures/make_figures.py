"""SVG figures for the write-up (light and dark mode). Usage: python figures/make_figures.py [block freq tok drop arch loss]"""
import json
import math
import pickle
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE
OUT.mkdir(exist_ok=True)
FONT = "-apple-system,Segoe UI,Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,Consolas,Menlo,monospace"
STYLE = (
    "<style>.bg{fill:#fcfcfb}.t1{fill:#0b0b0b}.t2{fill:#52514e}.t3{fill:#7a7973}"
    ".grid{stroke:#e4e3df}.axis{stroke:#9c9b95}.s1{fill:#2a78d6}.s1s{stroke:#2a78d6}"
    ".s2{fill:#c96f2a}.s2s{stroke:#c96f2a}.mute{fill:#c9c8c2}.tile{fill:#f1f0ec;stroke:#c9c8c2}"
    ".band{fill:#2a78d6;fill-opacity:.10}.on{fill:#ffffff}"
    "@media (prefers-color-scheme:dark){.bg{fill:#1a1a19}.t1{fill:#f0efec}.t2{fill:#c3c2b7}.t3{fill:#8f8e88}"
    ".grid{stroke:#2e2e2b}.axis{stroke:#7a7973}.s1{fill:#3987e5}.s1s{stroke:#3987e5}"
    ".s2{fill:#c96f2a}.s2s{stroke:#c96f2a}.mute{fill:#4a4945}.tile{fill:#262624;stroke:#4a4945}"
    ".band{fill:#3987e5;fill-opacity:.14}.on{fill:#ffffff}}</style>"
)


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def svg(name, w, h, title, body):
    doc = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
           f'font-family="{FONT}">\n<title>{esc(title)}</title>\n{STYLE}\n'
           f'<rect class="bg" width="{w}" height="{h}" rx="8"/>\n' + "\n".join(body) + "\n</svg>\n")
    (OUT / name).write_text(doc, encoding="utf-8")
    print("wrote", name)


def text(x, y, s, cls="t1", size=12, anchor="start", weight=None, family=None):
    w = f' font-weight="{weight}"' if weight else ""
    f = f' font-family="{family}"' if family else ""
    return f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}"{w}{f}>{esc(s)}</text>'


def hbar(x, y, w, h, cls):
    """Horizontal bar, square at the baseline (left), 4px rounded data-end."""
    r = min(4, w / 2, h / 2)
    return (f'<path class="{cls}" d="M{x:.1f},{y:.1f} h{w - r:.1f} a{r},{r} 0 0 1 {r},{r} v{h - 2 * r:.1f} '
            f'a{r},{r} 0 0 1 -{r},{r} h-{w - r:.1f} z"/>')


def vbar(x, base, w, h, cls):
    """Vertical bar growing up from the baseline, rounded top."""
    r = min(4, w / 2, h / 2) if h > 0 else 0
    if h <= 0:
        return ""
    return (f'<path class="{cls}" d="M{x:.1f},{base:.1f} v-{h - r:.1f} a{r},{r} 0 0 1 {r},-{r} h{w - 2 * r:.1f} '
            f'a{r},{r} 0 0 1 {r},{r} v{h - r:.1f} z"/>')


def legend(x, y, items):
    out, cx = [], x
    for label, cls in items:
        out.append(f'<rect class="{cls}" x="{cx}" y="{y - 9}" width="10" height="10" rx="2"/>')
        out.append(text(cx + 16, y, label, "t2", 12))
        cx += 16 + 7.2 * len(label) + 22
    return out


# 1. One chunk, eight examples -------------------------------------------------
def block_window():
    meta = pickle.load(open(HERE.parent / "data/poet_char/meta.pkl", "rb"))
    stoi = meta["stoi"]
    chunk = "Los padre"
    ids = [stoi[c] for c in chunk]
    W, tile, gap = 760, 34, 6
    body = [text(24, 34, "One chunk of block_size + 1 characters holds block_size training examples", "t1", 17, weight=600),
            text(24, 56, "block_size = 8, the first nine characters of the training split", "t2", 12.5)]
    x0, y0 = 150, 84
    body.append(text(x0 - 14, y0 + 22, "chunk", "t2", 12, "end"))
    body.append(text(x0 - 14, y0 + 52, "integers", "t2", 12, "end"))
    for i, (c, n) in enumerate(zip(chunk, ids)):
        x = x0 + i * (tile + gap)
        body.append(f'<rect class="tile" x="{x}" y="{y0}" width="{tile}" height="{tile}" rx="5"/>')
        body.append(text(x + tile / 2, y0 + 23, "␣" if c == " " else c, "t1", 16, "middle", family=MONO))
        body.append(text(x + tile / 2, y0 + 52, str(n), "t2", 11.5, "middle", family=MONO))
    # x and y brackets
    xe = x0 + 8 * (tile + gap) - gap
    body.append(f'<line class="s1s" x1="{x0}" y1="{y0 - 8}" x2="{xe}" y2="{y0 - 8}" stroke-width="2"/>')
    body.append(text(x0, y0 - 14, "x: input", "t2", 11.5))
    ys, ye = x0 + (tile + gap), x0 + 9 * (tile + gap) - gap
    body.append(f'<line class="s2s" x1="{ys}" y1="{y0 + 64}" x2="{ye}" y2="{y0 + 64}" stroke-width="2"/>')
    body.append(text(ye, y0 + 80, "y: x shifted one to the left", "t2", 11.5, "end"))
    # staircase
    sy, small, sg = 196, 22, 3
    body.append(text(24, sy - 12, "What the model is asked, position by position", "t1", 13, weight=600))
    for t in range(8):
        y = sy + t * (small + 7)
        body.append(text(x0 - 14, y + 15, f"t = {t}", "t2", 11.5, "end", family=MONO))
        for i in range(t + 1):
            x = x0 + i * (small + sg)
            body.append(f'<rect class="s1" x="{x}" y="{y}" width="{small}" height="{small}" rx="4"/>')
            body.append(text(x + small / 2, y + 15.5, "␣" if chunk[i] == " " else chunk[i], "on", 12.5, "middle", family=MONO))
        x = x0 + (t + 1) * (small + sg) + 14
        body.append(text(x - 7, y + 15, "›", "t3", 14, "middle"))
        body.append(f'<rect class="s2" x="{x + 2}" y="{y}" width="{small}" height="{small}" rx="4"/>')
        body.append(text(x + 2 + small / 2, y + 15.5, "␣" if chunk[t + 1] == " " else chunk[t + 1], "on", 12.5, "middle", family=MONO))
    body += legend(470, sy + 30, [("context the model sees", "s1")])
    body += legend(470, sy + 52, [("character it must predict", "s2")])
    body.append(text(470, sy + 88, "All eight are computed in one forward", "t2", 12))
    body.append(text(470, sy + 106, "pass: the causal mask hides the future,", "t2", 12))
    body.append(text(470, sy + 124, "so position t only ever sees 0 … t.", "t2", 12))
    svg("block-window.svg", W, sy + 8 * (small + 7) + 18, "One chunk, eight training examples", body)


# 2. Character frequencies ------------------------------------------------------
def char_frequency():
    data = json.load(open(HERE / "char_counts.json", encoding="utf-8"))
    counts = data["counts"]
    W, H = 760, 330
    L, R, T, B = 64, 24, 84, 270
    pw = W - L - R
    slot = pw / len(counts)
    bw = max(1.2, slot - 1.0)
    ymax = 6
    y = lambda v: B - (math.log10(v) / ymax) * (B - T)
    body = [text(24, 34, "The vocabulary is the characters that appear, and a long tail appears almost never", "t1", 16, weight=600),
            text(24, 55, "Occurrences of each of the 202 distinct characters, most frequent first (log scale)", "t2", 12.5)]
    for k in range(0, ymax + 1):
        yy = y(10 ** k)
        body.append(f'<line class="grid" x1="{L}" y1="{yy:.1f}" x2="{W - R}" y2="{yy:.1f}" stroke-width="1"/>')
        lab = {0: "1", 1: "10", 2: "100", 3: "1K", 4: "10K", 5: "100K", 6: "1M"}[k]
        body.append(text(L - 8, yy + 4, lab, "t2", 11, "end"))
    for i, c in enumerate(counts):
        h = B - y(c) if c > 1 else 2
        body.append(vbar(L + i * slot, B, bw, max(h, 2), "s1" if c >= 5 else "mute"))
    body.append(f'<line class="axis" x1="{L}" y1="{B}" x2="{W - R}" y2="{B}" stroke-width="1"/>')
    # threshold
    yt = y(5)
    body.append(f'<line class="s2s" x1="{L}" y1="{yt:.1f}" x2="{W - R}" y2="{yt:.1f}" stroke-width="1.5" stroke-dasharray="4 3"/>')
    body.append(text(W - R, yt - 6, "cut: fewer than 5 occurrences", "t2", 11.5, "end"))
    cut = sum(c >= 5 for c in counts)
    xc = L + cut * slot
    body.append(text(L, y(598402) - 8, "space 598,402 times, then e, a, o, s", "t2", 11.5))
    body.append(text(xc + 4, B + 18, f"{len(counts) - cut} removed", "t2", 11.5))
    body.append(text(xc - 4, B + 18, f"{cut} kept", "t2", 11.5, "end"))
    body += legend(L, H - 18, [("kept in the vocabulary", "s1"), ("removed before encoding", "mute")])
    svg("char-frequency.svg", W, H, "Character frequencies in the corpus", body)


# 3. Tokenizers -----------------------------------------------------------------
def tokenizers():
    rows = [("Characters", 122, 1.00, "s1"), ("SentencePiece, 1K", 1000, 2.91, "s1"),
            ("SentencePiece, 4K", 4000, 3.80, "s1"), ("SentencePiece, 8K", 8000, 4.17, "s1"),
            ("GPT-2 BPE (tiktoken)", 50257, 2.64, "s2")]
    W, H = 760, 300
    body = [text(24, 34, "A vocabulary learned on this text compresses it better than GPT-2's, at a fraction of the size", "t1", 15.5, weight=600),
            text(24, 55, "Measured on the 370,539-character validation split; embedding rows at the model's width of 256", "t2", 12.5)]
    lab_x, top, rh = 24, 92, 34
    panels = [(200, 230, "Characters per token", lambda r: r[2], 4.5, lambda v: f"{v:.2f}"),
              (490, 230, "Embedding parameters", lambda r: r[1] * 256, 13_000_000,
               lambda v: f"{v / 1e6:.1f}M" if v >= 1e6 else f"{v / 1e3:.0f}K")]
    for i, r in enumerate(rows):
        yy = top + i * rh
        body.append(text(lab_x, yy + 15, r[0], "t1", 12.5))
    for px, pw, title, f, vmax, fmt in panels:
        body.append(text(px, top - 14, title, "t2", 12, weight=600))
        body.append(f'<line class="axis" x1="{px}" y1="{top - 6}" x2="{px}" y2="{top + len(rows) * rh - 10}" stroke-width="1"/>')
        for i, r in enumerate(rows):
            v = f(r)
            w = max(3, v / vmax * pw)
            yy = top + i * rh
            body.append(hbar(px, yy + 2, w, 18, r[3]))
            body.append(text(px + w + 6, yy + 15.5, fmt(v), "t2", 11.5))
    body += legend(24, H - 20, [("trained on the poet's own text", "s1"), ("GPT-2's vocabulary, learned mostly on English", "s2")])
    svg("tokenizers.svg", W, H, "Tokenizers compared on the corpus", body)


# 4. The dropout trap -----------------------------------------------------------
def dropout_trap():
    rows = [("No dropout", 1664, "s1"), ("Dropout 0.2 everywhere", 3751, "s2"),
            ("Dropout 0.2, none inside attention", 1656, "s1")]
    W, H = 760, 236
    body = [text(24, 34, "Dropout inside attention costs more than the rest of the training step", "t1", 16, weight=600),
            text(24, 55, "One training step of the 4.75M model on the CPU, batch 32 × 256 characters, mean of 5 steps", "t2", 12.5)]
    px, pw, top, rh = 270, 400, 84, 40
    for i, (lab, v, cls) in enumerate(rows):
        yy = top + i * rh
        body.append(text(24, yy + 16, lab, "t1", 12.5))
        w = v / 4000 * pw
        body.append(hbar(px, yy + 3, w, 20, cls))
        body.append(text(px + w + 6, yy + 17, f"{v:,} ms", "t2", 11.5))
    body.append(f'<line class="axis" x1="{px}" y1="{top - 4}" x2="{px}" y2="{top + 3 * rh - 8}" stroke-width="1"/>')
    body += legend(24, H - 20, [("fused attention kernel", "s1"), ("generic fallback (math) kernel", "s2")])
    svg("dropout-trap.svg", W, H, "The cost of attention dropout on a CPU", body)


# 6. Architecture ---------------------------------------------------------------
def architecture():
    W, H = 760, 640
    body = [text(24, 34, "The model: a six-block transformer that reads 256 characters and scores the next one", "t1", 15.5, weight=600),
            text(24, 55, "Shapes for one sequence; parameter count on the right, row by row", "t2", 12.5),
            '<defs><marker id="ah" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto">'
            '<path class="t3" d="M0,0 L8,4 L0,8 z"/></marker></defs>']

    def box(x, y, w, h, title, sub=None, cls="tile"):
        out = [f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="7"/>']
        if sub:
            out.append(text(x + w / 2, y + h / 2 - 3, title, "t1", 13, "middle", weight=600))
            out.append(text(x + w / 2, y + h / 2 + 13, sub, "t2", 11.5, "middle"))
        else:
            out.append(text(x + w / 2, y + h / 2 + 4.5, title, "t1", 13, "middle", weight=600))
        return out

    def arrow(x1, y1, x2, y2):
        return f'<line class="axis" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke-width="1.5" marker-end="url(#ah)"/>'

    def plus(cx, cy):
        return [f'<circle class="bg" cx="{cx}" cy="{cy}" r="10"/>',
                f'<circle class="s1s" cx="{cx}" cy="{cy}" r="10" fill="none" stroke-width="2"/>',
                f'<line class="s1s" x1="{cx - 5}" y1="{cy}" x2="{cx + 5}" y2="{cy}" stroke-width="2"/>',
                f'<line class="s1s" x1="{cx}" y1="{cy - 5}" x2="{cx}" y2="{cy + 5}" stroke-width="2"/>']

    ledger = []  # (y, label, value)
    # input
    chunk, ids = "Los padre", "39 71 75 1 72 57 60 74 61"
    t0 = 250 - 9 * 26 / 2
    for i, c in enumerate(chunk):
        body.append(f'<rect class="tile" x="{t0 + i * 26:.1f}" y="74" width="22" height="22" rx="4"/>')
        body.append(text(t0 + i * 26 + 11, 89.5, "␣" if c == " " else c, "t1", 12.5, "middle", family=MONO))
    body.append(text(t0 + 9 * 26 + 4, 89.5, "…", "t2", 13))
    body.append(text(250, 116, ids + " …", "t2", 11.5, "middle", family=MONO))
    body.append(text(t0 - 10, 89.5, "256 characters", "t2", 11.5, "end"))
    body.append(text(t0 - 10, 116, "256 integers", "t2", 11.5, "end"))
    # embeddings
    body += box(100, 136, 140, 46, "Token table", "122 × 256")
    body += box(260, 136, 140, 46, "Position table", "256 × 256")
    body.append(arrow(170, 122, 170, 134))
    body.append(text(330, 129, "positions 0 … 255", "t3", 10.5, "middle"))
    ledger += [(159, "token table", "31,232"), (176, "position table", "65,536*")]
    body.append(f'<path class="axis" d="M170,182 V200 H238" fill="none" stroke-width="1.5" marker-end="url(#ah)"/>')
    body.append(f'<path class="axis" d="M330,182 V200 H262" fill="none" stroke-width="1.5" marker-end="url(#ah)"/>')
    body += plus(250, 200)
    # block group
    gx, gy, gw, gh = 84, 228, 332, 214
    body.append(f'<rect class="outline" x="{gx}" y="{gy}" width="{gw}" height="{gh}" rx="10" fill="none" '
                f'stroke-dasharray="5 4" style="stroke:#9c9b95"/>')
    body.append(text(gx + gw - 12, gy + gh - 12, "× 6 blocks", "t1", 12.5, "end", weight=600))
    sx = 140
    body.append(f'<line class="s1s" x1="250" y1="210" x2="250" y2="218" stroke-width="3"/>')
    body.append(f'<path class="s1s" d="M250,218 H{sx} V{gy + gh + 30}" fill="none" stroke-width="3"/>')
    body.append(text(sx - 8, gy + 40, "residual", "t2", 11, "end"))
    body.append(text(sx - 8, gy + 54, "stream", "t2", 11, "end"))
    # attention branch
    body.append(f'<path class="axis" d="M{sx},258 H192" fill="none" stroke-width="1.5" marker-end="url(#ah)"/>')
    body += box(194, 240, 206, 50, "Causal self-attention", "layer norm first · 8 heads of 32")
    body.append(f'<path class="axis" d="M297,290 V312 H{sx + 12}" fill="none" stroke-width="1.5" marker-end="url(#ah)"/>')
    body += plus(sx, 312)
    # mlp branch
    body.append(f'<path class="axis" d="M{sx},346 H192" fill="none" stroke-width="1.5" marker-end="url(#ah)"/>')
    body += box(194, 328, 206, 50, "MLP", "layer norm first · 256, 1,024, 256")
    body.append(f'<path class="axis" d="M297,378 V400 H{sx + 12}" fill="none" stroke-width="1.5" marker-end="url(#ah)"/>')
    body += plus(sx, 400)
    ledger += [(265, "attention, 4 × 256²", "262,144"), (353, "MLP, 2 × 256 × 1,024", "524,288"),
               (420, "six blocks", "4,718,592")]
    # head
    body.append(f'<path class="s1s" d="M{sx},{gy + gh + 30} H236" fill="none" stroke-width="3"/>')
    body.append(arrow(236, gy + gh + 30, 238, gy + gh + 30))
    body += box(240, gy + gh + 12, 160, 36, "Final layer norm")
    body.append(arrow(320, gy + gh + 48, 320, gy + gh + 64))
    hy = gy + gh + 66
    body += box(240, hy, 160, 46, "Output head", "256 × 122")
    ledger += [(gy + gh + 34, "layer norms (13 × 256)", "3,328"), (hy + 27, "output head (tied)", "0")]
    # tied weights arc
    body.append(f'<path class="s2s" d="M238,{hy + 23} H74 Q60,{hy + 23} 60,{hy + 9} V173 Q60,159 74,159 H98" fill="none" stroke-width="1.5" stroke-dasharray="4 3"/>')
    body.append(text(74, hy + 40, "tied: the head reuses", "t2", 11))
    body.append(text(74, hy + 54, "the token table's weights", "t2", 11))
    # softmax output
    body.append(arrow(320, hy + 46, 320, hy + 62))
    oy = hy + 66
    body.append(text(250, oy + 12, "softmax: 122 probabilities for the next character", "t2", 11.5, "middle"))
    probs = json.load(open(HERE / "next_char.json", encoding="utf-8")) if (HERE / "next_char.json").exists() else None
    if probs:
        body.append(text(250, oy + 30, f"after “{probs['context']}”, the trained model gives", "t3", 10.5, "middle"))
        bx = 250 - len(probs["top"]) * 46 / 2
        for i, (c, p) in enumerate(probs["top"]):
            h = max(2, p * 46)
            body.append(vbar(bx + i * 46 + 12, oy + 92, 22, h, "s1" if i == 0 else "mute"))
            body.append(text(bx + i * 46 + 23, oy + 106, "␣" if c == " " else c, "t1", 12, "middle", family=MONO))
            body.append(text(bx + i * 46 + 23, oy + 88 - h, f"{p:.2f}", "t2", 10.5, "middle"))
    # ledger
    lx, vx = 470, 736
    body.append(text(lx, 74, "Parameters", "t1", 12.5, weight=600))
    body.append(f'<line class="grid" x1="{lx}" y1="80" x2="{vx}" y2="80" stroke-width="1"/>')
    for y, lab, val in ledger:
        body.append(text(lx, y, lab, "t2", 11.5))
        body.append(text(vx, y, val, "t1", 11.5, "end", family=MONO))
    ty = hy + 60
    body.append(f'<line class="axis" x1="{lx}" y1="{ty - 14}" x2="{vx}" y2="{ty - 14}" stroke-width="1"/>')
    body.append(text(lx, ty + 2, "total, as nanoGPT counts it", "t1", 12, weight=600))
    body.append(text(vx, ty + 2, "4,753,152", "t1", 12, "end", weight=600, family=MONO))
    body.append(text(lx, ty + 22, "* the position table is left out of the count", "t3", 10.5))
    svg("architecture.svg", W, H + 64 if probs else H, "The model architecture", body)


# 5. Loss curve -----------------------------------------------------------------
def loss_curve():
    log = (HERE.parent / "logs" / "train.log").read_text(encoding="utf-8", errors="replace")
    pts = [(int(a), float(b), float(c)) for a, b, c in
           re.findall(r"step (\d+): train loss ([\d.]+), val loss ([\d.]+)", log)]
    if len(pts) < 3:
        print("loss curve: not enough points yet"); return
    W, H = 760, 380
    L, R, T, B = 60, 150, 74, 320
    xmax, ymin, ymax = 3000, 1.0, 5.0
    X = lambda s: L + s / xmax * (W - L - R)
    Y = lambda v: B - (v - ymin) / (ymax - ymin) * (B - T)
    body = [text(24, 34, "Loss on the posts the model never trains on, against the predictions made before training", "t1", 15.5, weight=600),
            text(24, 55, "Cross-entropy in nats per character, measured every 250 steps on 50 batches of each split", "t2", 12.5)]
    for v in (1, 2, 3, 4, 5):
        body.append(f'<line class="grid" x1="{L}" y1="{Y(v):.1f}" x2="{W - R}" y2="{Y(v):.1f}" stroke-width="1"/>')
        body.append(text(L - 8, Y(v) + 4, f"{v:.1f}", "t2", 11, "end"))
    for s in range(0, xmax + 1, 500):
        body.append(text(X(s), B + 18, f"{s:,}", "t2", 11, "middle"))
    body.append(f'<line class="axis" x1="{L}" y1="{B}" x2="{W - R}" y2="{B}" stroke-width="1"/>')
    body.append(text((L + W - R) / 2, B + 38, "training step", "t2", 11.5, "middle"))
    # prediction band and reference lines
    body.append(f'<rect class="band" x="{L}" y="{Y(1.55):.1f}" width="{W - L - R}" height="{Y(1.35) - Y(1.55):.1f}"/>')
    body.append(text(W - R + 8, Y(1.45) + 4, "predicted 1.35–1.55", "t2", 11))
    yl = Y(math.log(122))
    body.append(f'<line class="axis" x1="{L}" y1="{yl:.1f}" x2="{W - R}" y2="{yl:.1f}" stroke-width="1" stroke-dasharray="3 3"/>')
    body.append(text(W - R + 8, yl + 4, "ln 122 = 4.80", "t2", 11))
    for idx, cls in ((1, "s2s"), (2, "s1s")):
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(p[0]):.1f},{Y(p[idx]):.1f}" for i, p in enumerate(pts))
        body.append(f'<path class="{cls}" d="{d}" fill="none" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
    last = pts[-1]
    for idx, cls, lab in ((2, "s1", f"val {last[2]:.2f}"), (1, "s2", f"train {last[1]:.2f}")):
        cx, cy = X(last[0]), Y(last[idx])
        body.append(f'<circle class="bg" cx="{cx:.1f}" cy="{cy:.1f}" r="6"/><circle class="{cls}" cx="{cx:.1f}" cy="{cy:.1f}" r="4"/>')
    gap = Y(last[2]) - Y(last[1])
    ytr, yva = Y(last[1]), Y(last[2])
    if abs(gap) < 14:
        ytr, yva = (min(ytr, yva) - 7, max(ytr, yva) + 7) if last[1] > last[2] else (max(ytr, yva) + 7, min(ytr, yva) - 7)
    body.append(text(X(last[0]) + 10, yva + 4, f"val {last[2]:.2f}", "t1", 11.5))
    body.append(text(X(last[0]) + 10, ytr + 4, f"train {last[1]:.2f}", "t2", 11.5))
    body += legend(L, H - 12, [("validation (posts held out)", "s1"), ("training", "s2")])
    svg("loss-curve.svg", W, H, "Training and validation loss", body)


if __name__ == "__main__":
    which = sys.argv[1:] or ["block", "freq", "tok", "drop", "arch", "loss"]
    for k, fn in {"block": block_window, "freq": char_frequency, "tok": tokenizers,
                  "drop": dropout_trap, "arch": architecture, "loss": loss_curve}.items():
        if k in which:
            fn()
