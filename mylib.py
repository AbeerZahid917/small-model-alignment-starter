import torch

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