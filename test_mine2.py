import math
import torch
from mylib import compute_response_logprobs

class Tok:
    eos_token_id = 2
    pad_token_id = 0
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

class NoEosTok(Tok):
    eos_token_id = None
    pad_token_id = None

class Out:
    pass

class UniformModel(torch.nn.Module):
    """All logits equal, so every token has probability 1/V."""
    def __init__(self, vocab=200):
        super().__init__()
        self.vocab = vocab
        self.dummy = torch.nn.Parameter(torch.zeros(1))
    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        out = Out()
        out.logits = torch.zeros(B, T, self.vocab) + self.dummy
        return out

class NextTokenModel(torch.nn.Module):
    """Strongly predicts that the next token is (current token + 1)."""
    def __init__(self, vocab=200):
        super().__init__()
        self.vocab = vocab
        self.dummy = torch.nn.Parameter(torch.zeros(1))
    def forward(self, input_ids, attention_mask=None):
        B, T = input_ids.shape
        logits = torch.zeros(B, T, self.vocab)
        for b in range(B):
            for t in range(T):
                logits[b, t, (int(input_ids[b, t]) + 1) % self.vocab] = 50.0
        out = Out()
        out.logits = logits + self.dummy
        return out

# --- masking test: uniform model, only response tokens may count ---
m = UniformModel()
lp = compute_response_logprobs(m, Tok(), ["ab"], ["cd"], 10)       # response c, d, EOS = 3 tokens
assert abs(lp.item() - (-3 * math.log(200))) < 1e-4
lp = compute_response_logprobs(m, Tok(), ["ab"], ["cd"], 4)        # EOS truncated -> 2 tokens
assert abs(lp.item() - (-2 * math.log(200))) < 1e-4
lp = compute_response_logprobs(m, Tok(), ["ab", "x"], ["cd", "y"], 10)   # padding must not count
assert lp.shape == (2,)
assert abs(lp[0].item() - (-3 * math.log(200))) < 1e-4
assert abs(lp[1].item() - (-2 * math.log(200))) < 1e-4             # y, EOS

# --- shift test: ids 97,98,99,100; model predicts previous+1; response = 99,100 ---
lp = compute_response_logprobs(NextTokenModel(), NoEosTok(), ["ab"], ["cd"], 10)
assert lp.item() > -1e-3, "log-prob should be about 0; check the logits/labels shift"
print("all good")