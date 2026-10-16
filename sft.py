import argparse
import json
import math
import os
import random
import shutil
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mylib import (build_sft_batch, format_prompt, read_jsonl, sft_loss_from_batch, token_logps)

p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
p.add_argument("--train", default="data/processed/sft_train.jsonl")
p.add_argument("--val", default="data/processed/sft_val.jsonl")
p.add_argument("--out", default="outputs/sft")
p.add_argument("--device", default="cuda:0")
p.add_argument("--max_len", type=int, default=512)
p.add_argument("--bs", type=int, default=2)
p.add_argument("--accum", type=int, default=16)
p.add_argument("--lr", type=float, default=2e-5)
p.add_argument("--warmup_frac", type=float, default=0.03)
p.add_argument("--wd", type=float, default=0.1)
p.add_argument("--clip", type=float, default=1.0)
p.add_argument("--epochs", type=int, default=1)
p.add_argument("--eval_every", type=int, default=100)
p.add_argument("--max_train", type=int, default=0)
p.add_argument("--bf16", action="store_true")
p.add_argument("--grad_ckpt", action="store_true")
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()


amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
os.makedirs(args.out, exist_ok=True)
device = args.device


last_dir = os.path.join(args.out, "last")
resume_step = 0
best_val = float("inf")
load_from = args.model
if os.path.exists(os.path.join(last_dir, "state.json")):
    state = json.load(open(os.path.join(last_dir, "state.json")))
    resume_step = state["opt_step"]
    best_val = state["best_val"]
    load_from = last_dir
    print(f"RESUMING from optimizer step {resume_step}, best_val {best_val}")


tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(load_from, torch_dtype=torch.float32).to(device)
model.config.use_cache = False
if args.grad_ckpt:
    model.gradient_checkpointing_enable()
model.train()


train_rows = read_jsonl(args.train)
val_rows = read_jsonl(args.val)
if args.max_train > 0:
    train_rows = train_rows[:args.max_train]
    val_rows = val_rows[:100]




def collate(rows):
    prompts = []
    responses = []
    for r in rows:
        prompts.append(format_prompt(r["prompt"]))
        responses.append(r["response"])
    return build_sft_batch(tok, prompts, responses, args.max_len)




all_batches = []
for epoch in range(args.epochs):
    order = list(range(len(train_rows)))
    random.Random(args.seed + epoch).shuffle(order)
    for i in range(0, len(order) - args.bs + 1, args.bs):
        all_batches.append(order[i:i + args.bs])
total_steps = len(all_batches) // args.accum
all_batches = all_batches[:total_steps * args.accum]
warmup_steps = max(1, int(args.warmup_frac * total_steps))
print(f"micro-batches: {len(all_batches)}, optimizer steps: {total_steps}, warmup: {warmup_steps}")




def lr_factor(step):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress))




decay, no_decay = [], []
for name, param in model.named_parameters():
    if param.ndim >= 2:
        decay.append(param)
    else:
        no_decay.append(param)
optimizer = torch.optim.AdamW(
    [{"params": decay, "weight_decay": args.wd}, {"params": no_decay, "weight_decay": 0.0}],
    lr=args.lr, betas=(0.9, 0.95))
scaler = torch.amp.GradScaler("cuda", enabled=not args.bf16)
metrics_path = os.path.join(args.out, "metrics.jsonl")




def log(rec):
    with open(metrics_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(rec, flush=True)




@torch.no_grad()
def evaluate():
    model.eval()
    total_nll = 0.0
    total_tokens = 0.0
    for i in range(0, len(val_rows), 4):
        batch = collate(val_rows[i:i + 4])
        with torch.autocast("cuda", dtype=amp_dtype):
            lp, mask = token_logps(model, batch)
        total_nll += float((-lp * mask).sum())
        total_tokens += float(mask.sum())
    model.train()
    return total_nll / max(1.0, total_tokens)




def save_last(opt_step):
    tmp = os.path.join(args.out, "last_tmp")
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    model.save_pretrained(tmp)
    json.dump({"opt_step": opt_step, "best_val": best_val}, open(os.path.join(tmp, "state.json"), "w"))
    if os.path.exists(last_dir):
        shutil.rmtree(last_dir)
    os.rename(tmp, last_dir)




running = 0.0
t0 = time.time()
for micro in range(resume_step * args.accum, len(all_batches)):
    rows = []
    for j in all_batches[micro]:
        rows.append(train_rows[j])
    batch = collate(rows)

    with torch.autocast("cuda", dtype=amp_dtype):
        loss = sft_loss_from_batch(model, batch)
    scaler.scale(loss / args.accum).backward()
    running += float(loss)

    if (micro + 1) % args.accum == 0:
        opt_step = (micro + 1) // args.accum
        lr = args.lr * lr_factor(opt_step - 1)
        for group in optimizer.param_groups:
            group["lr"] = lr
        scaler.unscale_(optimizer)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if opt_step % 10 == 0:
            log({"step": opt_step, "train_loss": running / (10 * args.accum), "lr": lr,
                 "grad_norm": float(gnorm), "elapsed_min": (time.time() - t0) / 60})
            running = 0.0

        if opt_step % args.eval_every == 0 or opt_step == total_steps:
            val = evaluate()
            log({"step": opt_step, "val_loss": val})
            if val < best_val:
                best_val = val
                model.save_pretrained(os.path.join(args.out, "best"))
                tok.save_pretrained(os.path.join(args.out, "best"))
            save_last(opt_step)




model.save_pretrained(os.path.join(args.out, "final"))
tok.save_pretrained(os.path.join(args.out, "final"))
print("done. best val loss:", best_val)
