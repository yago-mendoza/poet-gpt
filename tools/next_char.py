"""Usage: python tools/next_char.py "Los padres "
Top-5 next-character probabilities of the trained model for one context (for the architecture figure)."""
import json, pickle, sys, torch
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from model import GPT, GPTConfig
ck = torch.load("out-poet-char/ckpt.pt", map_location="cpu")
m = GPT(GPTConfig(**ck["model_args"])); sd = ck["model"]
m.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in sd.items()}); m.eval()
meta = pickle.load(open("data/poet_char/meta.pkl", "rb"))
ctx = sys.argv[1]
x = torch.tensor([[meta["stoi"][c] for c in ctx]])
with torch.no_grad():
    logits, _ = m(x)
p = torch.softmax(logits[0, -1], -1)
top = torch.topk(p, 5)
res = {"context": ctx, "top": [(meta["itos"][int(i)], round(float(v), 3)) for v, i in zip(top.values, top.indices)]}
print(res)
json.dump(res, open("figures/next_char.json", "w", encoding="utf-8"), ensure_ascii=False)
print("best val loss in checkpoint:", ck["best_val_loss"].item(), "iter", ck["iter_num"])
