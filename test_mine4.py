import math
import torch
from mylib import compute_dpo_loss_from_logps, compute_dpo_loss

# 1. equal logps -> loss = ln 2
v = torch.tensor([-2.0, -5.0])
assert abs(compute_dpo_loss_from_logps(v, v, v, v, 0.2).item() - math.log(2)) < 1e-6

# 2. hand-computed: policy gap 2, reference gap 0, beta 0.5 -> logits = 1
pc, pr = torch.tensor([-1.0]), torch.tensor([-3.0])
rc, rr = torch.tensor([-2.0]), torch.tensor([-2.0])
expected = -math.log(1 / (1 + math.exp(-1.0)))
assert abs(compute_dpo_loss_from_logps(pc, pr, rc, rr, 0.5).item() - expected) < 1e-5

# 3. policy better than reference -> loss below ln 2; policy worse -> above
better = compute_dpo_loss_from_logps(pc, pr, rc, rr, 0.5).item()
worse = compute_dpo_loss_from_logps(pr, pc, rc, rr, 0.5).item()
assert better < math.log(2) < worse

# 4. reference tensors must not get gradients, policy tensors must
pc2 = torch.tensor([-1.0], requires_grad=True)
pr2 = torch.tensor([-3.0], requires_grad=True)
rc2 = torch.tensor([-2.0], requires_grad=True)
rr2 = torch.tensor([-2.0], requires_grad=True)
compute_dpo_loss_from_logps(pc2, pr2, rc2, rr2, 0.5).backward()
assert pc2.grad is not None and pr2.grad is not None
assert rc2.grad is None and rr2.grad is None

# 5. full pipeline with two identical uniform models -> loss = ln 2, only policy gets gradient
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

policy, reference = UniformModel(), UniformModel()
loss = compute_dpo_loss(policy, reference, Tok(), ["ab", "cd"], ["x", "y"], ["z", "w"], 0.1, 10)
assert loss.dim() == 0
assert abs(loss.item() - math.log(2)) < 1e-5
loss.backward()
assert reference.dummy.grad is None
print("all good")