# Samples of 700 characters from an empty line at temperature 0.8, as for the character model.
# python sample_hf.py <modelo o carpeta> [--adapter carpeta LoRA] [--out fichero]
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument('model')
p.add_argument('--adapter')
p.add_argument('--out', default='samples.txt')
p.add_argument('--n', type=int, default=6)
p.add_argument('--chars', type=int, default=700)
args = p.parse_args()

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
tok = AutoTokenizer.from_pretrained(args.model)
# An explicit dtype: transformers 5 ignores torch_dtype and would load the checkpoint's float16, very slow on a CPU
model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16 if dev == 'cuda' else torch.float32).to(dev)
if args.adapter:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
model.eval()
torch.manual_seed(1337)
prompt = tok('\n', return_tensors='pt')['input_ids'].to(dev)
out = []
for i in range(args.n):
    with torch.no_grad():
        g = model.generate(prompt, do_sample=True, temperature=0.8, top_k=0, top_p=1.0,
                           max_new_tokens=args.chars // 3, pad_token_id=tok.eos_token_id)  # 700 characters at 3.6+ per token
    text = tok.decode(g[0, prompt.size(1):], skip_special_tokens=True)[:args.chars]
    out.append(text)
    print(f'--- {i + 1}\n{text}\n', flush=True)
open(args.out, 'w', encoding='utf-8').write('\n---------------\n'.join(out))
