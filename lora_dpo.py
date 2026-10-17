import argparse
import json

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


p = argparse.ArgumentParser()
p.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M-Instruct")
p.add_argument("--data", default="data/persona_lora/ramsay_style_preferences.json")
p.add_argument("--out", default="outputs/lora_dpo")
p.add_argument("--beta", type=float, default=0.1)
p.add_argument("--lr", type=float, default=2e-5)
p.add_argument("--epochs", type=int, default=3)
p.add_argument("--bs", type=int, default=4)
p.add_argument("--accum", type=int, default=4)
p.add_argument("--rank", type=int, default=16)
p.add_argument("--alpha", type=int, default=32)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()



rows = json.load(open(args.data))
conv = []
for r in rows:
    conv.append({
        "prompt": [{"role": "user", "content": r["prompt"]}],
        "chosen": [{"role": "assistant", "content": r["chosen"]}],
        "rejected": [{"role": "assistant", "content": r["rejected"]}],
    })
dataset = Dataset.from_list(conv)
print("examples:", len(dataset))



tok = AutoTokenizer.from_pretrained(args.model)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)



lora = LoraConfig(
    r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)



config = DPOConfig(
    output_dir=args.out, beta=args.beta, learning_rate=args.lr,
    num_train_epochs=args.epochs, per_device_train_batch_size=args.bs,
    gradient_accumulation_steps=args.accum, max_length=512, fp16=True,
    logging_steps=2, save_strategy="no", report_to="none", seed=args.seed,
)



trainer = DPOTrainer(model=model, ref_model=None, args=config, train_dataset=dataset,
                     processing_class=tok, peft_config=lora)
trainer.train()
trainer.model.save_pretrained(args.out + "/adapter")
tok.save_pretrained(args.out + "/adapter")

with open(args.out + "/log_history.json", "w") as f:
    json.dump(trainer.state.log_history, f, indent=1)
print("saved adapter + logs")