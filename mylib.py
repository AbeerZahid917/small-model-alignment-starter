import json
import torch
import torch.nn.functional as F

def format_prompt(instruction):
    return PROMPT_TEMPLATE.format(instruction=instruction.strip())




def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows




def encode_text(tokenizer, text):
    """Token ids without special tokens. Falls back for toy tokenizers."""
    if hasattr(tokenizer, "encode"):
        try:
            return list(tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return list(tokenizer.encode(text))
    return list(tokenizer(text)["input_ids"])




def build_sft_batch(tokenizer, prompts, responses, max_length):
    all_ids = []
    all_labels = []

    for (prompt, response) in zip(prompts, responses):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        response_ids = tokenizer.encode(response, add_special_tokens=False)

        if tokenizer.eos_token_id is not None:
            response_ids = response_ids + [tokenizer.eos_token_id]

        ids = prompt_ids + response_ids
        labels = [-100] * len(prompt_ids) + response_ids

        ids = ids[:max_length]
        labels = labels[:max_length]

        all_ids.append(ids)
        all_labels.append(labels)
    
    longest = 0
    for ids in all_ids:
        if len(ids) > longest:
            longest = len(ids)

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    input_ids = []
    attention_mask = []
    labels_out = []

    for (ids, labels) in zip(all_ids, all_labels):
        n_pad = longest - len(ids)
        input_ids.append(ids + [pad_id] * n_pad)
        attention_mask.append([1] * len(ids) + [0] * n_pad)
        labels_out.append(labels + [-100] * n_pad)
    
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels_out, dtype=torch.long),
    }




def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    chosen_batch = build_sft_batch(tokenizer, prompts, chosen_responses, max_length)
    rejected_batch = build_sft_batch(tokenizer, prompts, rejected_responses, max_length)
    return {"chosen": chosen_batch, "rejected": rejected_batch}




def token_logps(model, batch):
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    out = model(input_ids=input_ids, attention_mask=attention_mask) 
    logits = out.logits                      
    # [B, T, V]

    logits = logits[:, :-1, :].float()
    targets = labels[:, 1:]
    B, T, V = logits.shape
    neg_log_loss = F.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T), ignore_index=-100, reduction="none")
    neg_log_loss = neg_log_loss.view(B, T)
    logps = -neg_log_loss
    mask = (targets != -100).float()
    return logps, mask



def batch_response_logps(model, batch):
    logps, mask = token_logps(model, batch)
    return (logps * mask).sum(dim=1)




def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    return batch_response_logps(model, batch)




def sft_loss_from_batch(model, batch):
    logps, mask = token_logps(model, batch)
    total = (-logps * mask).sum()
    count = mask.sum().clamp(min=1.0)
    return total / count




def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    batch = build_sft_batch(tokenizer, prompts, responses, max_length)
    return sft_loss_from_batch(model, batch)




def compute_dpo_loss_from_logps(pc, pr, rc, rr, beta):
    policy = pc - pr
    reference = rc - rr

    reference = reference.detach()
    logits = beta * (policy - reference)

    per_example_loss = -torch.nn.functional.logsigmoid(logits)

    return per_example_loss.mean()




def compute_dpo_loss(policy_model, reference_model, tokenizer, prompts, chosen_responses, rejected_responses, beta, max_length):
    batches = build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
    chosen_batch = batches["chosen"]
    rejected_batch = batches["rejected"]

    pc = batch_response_logps(policy_model, chosen_batch)
    pr = batch_response_logps(policy_model, rejected_batch)

    with torch.no_grad():
        rc = batch_response_logps(reference_model, chosen_batch)
        rr = batch_response_logps(reference_model, rejected_batch)

    rc = rc.to(pc.device)
    rr = rr.to(pc.device)

    return compute_dpo_loss_from_logps(pc, pr, rc, rr, beta)




def dpo_step(policy_model, reference_model, tokenizer, prompts, chosen_responses,
             rejected_responses, beta, max_length):
    """Returns (loss, reward_chosen [B], reward_rejected [B]) - rewards are detached."""
    chosen_batch, rejected_batch = build_dpo_batch(
        tokenizer, prompts, chosen_responses, rejected_responses, max_length)

    pc = batch_response_logps(policy_model, chosen_batch)
    pr = batch_response_logps(policy_model, rejected_batch)
    with torch.no_grad():
        rc = batch_response_logps(reference_model, chosen_batch)
        rr = batch_response_logps(reference_model, rejected_batch)
    rc = rc.to(pc.device)
    rr = rr.to(pc.device)

    loss = compute_dpo_loss_from_logps(pc, pr, rc, rr, beta)
    reward_chosen = (beta * (pc - rc)).detach()
    reward_rejected = (beta * (pr - rr)).detach()
    return loss, reward_chosen, reward_rejected