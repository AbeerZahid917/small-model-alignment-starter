import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mylib import format_prompt

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--prompts", required=True)
p.add_argument("--out", required=True)
p.add_argument("--generator", required=True)
p.add_argument("--dataset", required=True)
p.add_argument("--max_new_tokens", type=int, default=256)
p.add_argument("--batch_size", type=int, default=16)
p.add_argument("--repetition_penalty", type=float, default=1.0)
p.add_argument("--no_repeat_ngram_size", type=int, default=0)
p.add_argument("--limit", type=int, default=0)
p.add_argument("--device", default="cuda:0")
args = p.parse_args()



tok = AutoTokenizer.from_pretrained(args.model)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(args.device)
model.eval()



raw = json.load(open(args.prompts))
items = []
for i, ex in enumerate(raw):
    text = ex.get("instruction") or ex.get("prompt")
    items.append({"id": str(ex.get("id", i)), "instruction": text})
if args.limit > 0:
    items = items[:args.limit]



results = {}
order = sorted(range(len(items)), key=lambda k: len(items[k]["instruction"]))
for start in range(0, len(order), args.batch_size):
    idxs = order[start:start + args.batch_size]
    prompts = [format_prompt(items[k]["instruction"]) for k in idxs]
    enc = tok(prompts, return_tensors="pt", padding=True).to(args.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                             repetition_penalty=args.repetition_penalty,
                             no_repeat_ngram_size=args.no_repeat_ngram_size,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    new_tokens = out[:, enc["input_ids"].shape[1]:]
    texts = tok.batch_decode(new_tokens, skip_special_tokens=True)
    for k, text in zip(idxs, texts):
        text = text.split("### Instruction")[0].strip()
        results[k] = text
    print(f"{start + len(idxs)}/{len(order)}", flush=True)



records = []
for k, item in enumerate(items):
    records.append({"id": item["id"], "instruction": item["instruction"], "output": results[k],
                    "generator": args.generator, "dataset": args.dataset})



os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w") as f:
    json.dump(records, f, indent=1)
print("saved", len(records), "->", args.out)
