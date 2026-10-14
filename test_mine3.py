import math
import torch
from mylib import compute_sft_loss

class Tok:
    eos_token_id = 2
    pad_token_id = 0
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

class Out:
    pass

class UniformModel(torch.nn.Module):
    def __init__(self, vocab=200):
        super().__init__()
        self.vocab = vocab
        self.dummy = torch.nn.Parameter(torch.zeros(1))
    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        out = Out()
        out.logits = torch.zeros(B, T, self.vocab) + self.dummy
        return out

m = UniformModel()
loss = compute_sft_loss(m, Tok(), ["ab", "x"], ["cd", "y"], 10)
assert loss.dim() == 0                                   # a single number
assert abs(loss.item() - math.log(200)) < 1e-4           # every token costs log(200)

loss.backward()
assert m.dummy.grad is not None                          # gradient reaches the model

loss = compute_sft_loss(m, Tok(), ["abcdef"], ["gh"], 3)  # everything masked -> must not be NaN
assert not torch.isnan(loss)
print("all good")