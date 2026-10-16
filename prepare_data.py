import json
import os
import random
from datasets import load_dataset

OUT = "data/processed"
SEED = 42
MAX_PROMPT_CHARS = 1000
PRINT_ROWS = True
os.makedirs(OUT, exist_ok=True)


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("wrote", len(rows), "->", path)



def first_turn(messages):
    if len(messages) < 2:
        return None
    if messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
        return None
    prompt = messages[0]["content"].strip()
    response = messages[1]["content"].strip()
    if len(prompt) == 0 or len(response) == 0 or len(prompt) > MAX_PROMPT_CHARS:
        return None
    return prompt, response




def build_sft(n_train=30000, n_val=1000):
    ds = load_dataset("HuggingFaceH4/ultrachat_200k")
    if PRINT_ROWS:
        print(ds["train_sft"][0]["messages"][:2])
    train = ds["train_sft"].shuffle(seed=SEED)
    test = ds["test_sft"].shuffle(seed=SEED)


    def collect(split, n):
        rows = []
        for ex in split:
            pair = first_turn(ex["messages"])
            if pair is None:
                continue
            rows.append({"prompt": pair[0], "response": pair[1]})
            if len(rows) >= n:
                break
        return rows

    write_jsonl(f"{OUT}/sft_train.jsonl", collect(train, n_train))
    write_jsonl(f"{OUT}/sft_val.jsonl", collect(test, n_val))




def parse_hh(text):
    if text.count("\n\nHuman:") != 1 or text.count("\n\nAssistant:") != 1:
        return None
    after_human = text.split("\n\nHuman:")[1]
    prompt, response = after_human.split("\n\nAssistant:")
    return prompt.strip(), response.strip()




def build_hh(n_train=15000, n_val=1000):
    ds = load_dataset("Anthropic/hh-rlhf")
    if PRINT_ROWS:
        print(repr(ds["train"][0]["chosen"][:300]))


    def collect(split, n):
        split = split.shuffle(seed=SEED)
        rows = []
        for ex in split:
            c = parse_hh(ex["chosen"])
            r = parse_hh(ex["rejected"])
            if c is None or r is None:
                continue
            if c[0] != r[0] or len(c[0]) > MAX_PROMPT_CHARS:
                continue
            if len(c[1]) == 0 or len(r[1]) == 0 or c[1] == r[1]:
                continue
            rows.append({"prompt": c[0], "chosen": c[1], "rejected": r[1]})
            if len(rows) >= n:
                break
        return rows

    write_jsonl(f"{OUT}/hh_train.jsonl", collect(ds["train"], n_train))
    write_jsonl(f"{OUT}/hh_val.jsonl", collect(ds["test"], n_val))




def build_uf(n_train=15000, n_val=1000):
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
    if PRINT_ROWS:
        ex = ds["train_prefs"][0]
        print(ex["prompt"][:200], "|", ex["chosen"], "|", ex["rejected"][-1]["content"][:100])


    def collect(split, n):
        split = split.shuffle(seed=SEED)
        rows = []
        for ex in split:
            if len(ex["chosen"]) != 2 or len(ex["rejected"]) != 2:
                continue
            prompt = ex["prompt"].strip()
            chosen = ex["chosen"][-1]["content"].strip()
            rejected = ex["rejected"][-1]["content"].strip()
            if len(prompt) > MAX_PROMPT_CHARS or len(chosen) == 0 or len(rejected) == 0:
                continue
            if chosen == rejected:
                continue
            rows.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
            if len(rows) >= n:
                break
        return rows

    write_jsonl(f"{OUT}/uf_train.jsonl", collect(ds["train_prefs"], n_train))
    write_jsonl(f"{OUT}/uf_val.jsonl", collect(ds["test_prefs"], n_val))




def build_alpaca(n=300):
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id="tatsu-lab/alpaca_eval", filename="alpaca_eval.json",
                           repo_type="dataset")
    data = json.load(open(path))
    rng = random.Random(SEED)
    idx = rng.sample(range(len(data)), n)
    subset = []
    for i in idx:
        subset.append({"id": f"alpaca_{i}", "instruction": data[i]["instruction"]})
    with open(f"{OUT}/alpaca_eval_{n}_seed{SEED}.json", "w") as f:
        json.dump(subset, f, indent=1)
    print("wrote alpaca subset", len(subset))



if __name__ == "__main__":
    build_alpaca()
    build_sft()
    build_hh()
    build_uf()
