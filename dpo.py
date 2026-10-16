import argparse
import json
import os
import random
import shutil
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from mylib import dpo_step, encode_text, format_prompt, read_jsonl



p = argparse.ArgumentParser()
p.add_argument("--sft_ckpt", default="outputs/sft/best")
p.add_argument("--train", required=True)
p.add_argument("--val", required=True)
p.add_argument("--out", required=True)
p.add_argument("--policy_device", default="cuda:0")
p.add_argument("--ref_device", default="cuda:0")
p.add_argument("--beta", type=float, default=0.1)
p.add_argument("--lr", type=float, default=1e-6)
p.add_argument("--accum", type=int, default=32)
p.add_argument("--clip", type=float, default=1.0)
p.add_argument("--max_len", type=int, default=512)
p.add_argument("--max_prompt_len", type=int, default=256)
p.add_argument("--eval_every", type=int, default=100)
p.add_argument("--eval_bs", type=int, default=4)
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




last_dir = os.path.join(args.out, "last")
resume_step = 0
best_acc = -1.0
policy_from = args.sft_ckpt
if os.path.exists(os.path.join(last_dir, "state.json")):
    state = json.load(open(os.path.join(last_dir, "state.json")))
    resume_step = state["opt_step"]
    best_acc = state["best_acc"]
    policy_from = last_dir
    print(f"RESUMING from optimizer step {resume_step}, best_acc {best_acc}")



tok = AutoTokenizer.from_pretrained(args.sft_ckpt)
policy = AutoModelForCausalLM.from_pretrained(policy_from, torch_dtype=torch.float32).to(args.policy_device)
policy.config.use_cache = False
if args.grad_ckpt:
    policy.gradient_checkpointing_enable()
policy.train()



reference = AutoModelForCausalLM.from_pretrained(args.sft_ckpt, torch_dtype=torch.float32).to(args.ref_device)
reference.eval()
for param in reference.parameters():
    param.requires_grad = False




def load_pairs(path):
    rows = read_jsonl(path)
    kept = []
    for r in rows:
        prompt = format_prompt(r["prompt"])
        if len(encode_text(tok, prompt)) <= args.max_prompt_len:   # filter, don't truncate
            kept.append({"prompt": prompt, "chosen": r["chosen"], "rejected": r["rejected"]})
    print(f"{path}: kept {len(kept)}/{len(rows)} (prompt <= {args.max_prompt_len} tokens)")
    return kept




train = load_pairs(args.train)
val = load_pairs(args.val)
if args.max_train > 0:
    train = train[:args.max_train]
    val = val[:100]
random.Random(args.seed).shuffle(train)        # fixed order -> resume continues exactly

optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=0.0)
scaler = torch.amp.GradScaler("cuda", enabled=not args.bf16)
metrics_path = os.path.join(args.out, "metrics.jsonl")




def log(rec):
    with open(metrics_path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(rec, flush=True)




@torch.no_grad()
def evaluate():
    policy.eval()
    loss_sum, acc_sum, margin_sum = 0.0, 0.0, 0.0
    for start in range(0, len(val), args.eval_bs):
        rows = val[start:start + args.eval_bs]
        with torch.autocast("cuda", dtype=amp_dtype):
            loss, rc, rr = dpo_step(policy, reference, tok,
                                    [r["prompt"] for r in rows],
                                    [r["chosen"] for r in rows],
                                    [r["rejected"] for r in rows],
                                    args.beta, args.max_len)
        loss_sum += float(loss) * len(rows)
        for i in range(len(rows)):
            if float(rc[i]) > float(rr[i]):
                acc_sum += 1.0
            margin_sum += float(rc[i] - rr[i])
    policy.train()
    n = len(val)
    return loss_sum / n, acc_sum / n, margin_sum / n




def save_last(opt_step):
    tmp = os.path.join(args.out, "last_tmp")
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    policy.save_pretrained(tmp)
    json.dump({"opt_step": opt_step, "best_acc": best_acc}, open(os.path.join(tmp, "state.json"), "w"))
    if os.path.exists(last_dir):
        shutil.rmtree(last_dir)
    os.rename(tmp, last_dir)




total_steps = len(train) // args.accum
print("total optimizer steps:", total_steps)




if resume_step == 0:
    v_loss, v_acc, v_margin = evaluate()
    log({"step": 0, "val_loss": v_loss, "val_reward_acc": v_acc, "val_reward_margin": v_margin})




win_loss, win_acc, win_margin, win_n = 0.0, 0.0, 0.0, 0
t0 = time.time()
for i in range(resume_step * args.accum, total_steps * args.accum):
    r = train[i]
    with torch.autocast("cuda", dtype=amp_dtype):
        loss, rc, rr = dpo_step(policy, reference, tok, [r["prompt"]], [r["chosen"]],
                                [r["rejected"]], args.beta, args.max_len)
    scaler.scale(loss / args.accum).backward()

    win_loss += float(loss)
    if float(rc[0]) > float(rr[0]):
        win_acc += 1.0
    win_margin += float(rc[0] - rr[0])
    win_n += 1

    if (i + 1) % args.accum == 0:
        opt_step = (i + 1) // args.accum
        scaler.unscale_(optimizer)
        gnorm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        log({"step": opt_step, "train_loss": win_loss / win_n, "train_reward_acc": win_acc / win_n,
             "train_reward_margin": win_margin / win_n, "grad_norm": float(gnorm),
             "elapsed_min": (time.time() - t0) / 60})
        win_loss, win_acc, win_margin, win_n = 0.0, 0.0, 0.0, 0

        if opt_step % args.eval_every == 0 or opt_step == total_steps:
            v_loss, v_acc, v_margin = evaluate()
            log({"step": opt_step, "val_loss": v_loss, "val_reward_acc": v_acc,
                 "val_reward_margin": v_margin})
            if v_acc > best_acc:
                best_acc = v_acc
                policy.save_pretrained(os.path.join(args.out, "best"))
                tok.save_pretrained(os.path.join(args.out, "best"))
            save_last(opt_step)




policy.save_pretrained(os.path.join(args.out, "final"))
tok.save_pretrained(os.path.join(args.out, "final"))
print("done. best val reward acc:", best_acc)
