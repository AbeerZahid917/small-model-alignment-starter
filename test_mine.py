from mylib import build_sft_batch

class FakeTok:
    eos_token_id = 2
    pad_token_id = 0
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]      # 'a' -> 97, 'b' -> 98 ...

class NoEosTok:
    eos_token_id = None
    pad_token_id = None
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

tok = FakeTok()

# 1. basic: prompt "ab", response "cd"
b = build_sft_batch(tok, ["ab"], ["cd"], 10)
assert b["input_ids"].tolist() == [[97, 98, 99, 100, 2]]
assert b["labels"].tolist() == [[-100, -100, 99, 100, 2]]
assert b["attention_mask"].tolist() == [[1, 1, 1, 1, 1]]

# 2. truncation: EOS is cut off
b = build_sft_batch(tok, ["ab"], ["cd"], 4)
assert b["input_ids"].tolist() == [[97, 98, 99, 100]]
assert b["labels"].tolist() == [[-100, -100, 99, 100]]

# 3. padding + row order
b = build_sft_batch(tok, ["ab", "x"], ["cd", "y"], 10)
assert b["input_ids"].tolist() == [[97, 98, 99, 100, 2], [120, 121, 2, 0, 0]]
assert b["attention_mask"].tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]
assert b["labels"].tolist() == [[-100, -100, 99, 100, 2], [-100, 121, 2, -100, -100]]

# 4. tokenizer with no EOS and no pad: must not crash
b = build_sft_batch(NoEosTok(), ["ab"], ["cd"], 10)
assert b["input_ids"].tolist() == [[97, 98, 99, 100]]

# 5. prompt longer than max_length: all labels -100, no crash
b = build_sft_batch(tok, ["abcdef"], ["gh"], 3)
assert b["labels"].tolist() == [[-100, -100, -100]]

print("all good")