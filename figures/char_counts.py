"""Occurrences of each character after removing mentions and hashtags (before the rare-character cut)."""
import json, re, sys
from collections import Counter
from pathlib import Path
mention, hashtag = re.compile(r"@[\w.]*\w"), re.compile(r"#\w+")
def clean(t):
    t = hashtag.sub("", mention.sub("", t)).replace("\xa0", " ")
    t = re.sub(r"[ \t]+", " ", t); t = re.sub(r" +([.,;:!?)])", r"\1", t)
    t = "\n".join(l.strip() for l in t.splitlines()); return re.sub(r"\n{3,}", "\n\n", t).strip()
posts = [clean(f.read_text(encoding="utf-8", errors="replace")) for f in sorted(Path(sys.argv[1]).rglob("*.txt"))]
freq = Counter("\n\n".join(p for p in posts if p))
counts = sorted(freq.values(), reverse=True)
print(len(counts), "distinct;", sum(c < 5 for c in counts), "under 5;", counts[:5], counts[-5:])
top = [(c, n) for c, n in freq.most_common(12)]
print("top:", [(repr(c), n) for c, n in top])
json.dump({"counts": counts, "top": top}, open("figures/char_counts.json", "w", encoding="utf-8"), ensure_ascii=False)
