import argparse
import json
import random

p = argparse.ArgumentParser()
p.add_argument("--dpo", required=True)
p.add_argument("--sft", required=True)
p.add_argument("--out", required=True)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

dpo = {r["id"]: r for r in json.load(open(args.dpo))}
sft = {r["id"]: r for r in json.load(open(args.sft))}
rng = random.Random(args.seed)



with open(args.out, "w") as f:
    for pid in dpo:
        if pid not in sft:
            continue
        if rng.random() < 0.5:
            a, b, hidden = dpo[pid]["output"], sft[pid]["output"], {"A": "dpo", "B": "sft"}
        else:
            a, b, hidden = sft[pid]["output"], dpo[pid]["output"], {"A": "sft", "B": "dpo"}
        rec = {"id": pid, "instruction": dpo[pid]["instruction"],
               "response_a": a, "response_b": b, "hidden": hidden}
        f.write(json.dumps(rec) + "\n")
        
print("wrote pairs ->", args.out)