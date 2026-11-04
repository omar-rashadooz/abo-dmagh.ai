"""Chapter 4 (practical): causal self-attention, one head, from scratch.

Idea: every token looks at all the tokens BEFORE it and decides who matters.
    Query = "what am I looking for?"
    Key   = "what do I contain?"
    Value = "what do I hand over if you pick me?"
Steps:  scores = Query . Key  ->  hide the future  ->  softmax  ->  weighted sum of Values

Run:  python ch4_self_attention.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)


class Head(nn.Module):
    """One head of causal self-attention (we will reuse this class in chapter 5)."""

    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # lower-triangular matrix of ones: row t may look at columns 0..t only
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x, return_weights=False):
        B, T, C = x.shape
        k = self.key(x)      # (B, T, head_size)
        q = self.query(x)    # (B, T, head_size)
        v = self.value(x)    # (B, T, head_size)

        # 1) similarity of every query with every key; the scaling keeps the numbers small
        wei = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5      # (B, T, T)
        # 2) no peeking at the future
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        # 3) softmax: each row becomes percentages that sum to 1
        wei = F.softmax(wei, dim=-1)
        # 4) mix the Values using those percentages
        out = wei @ v                                             # (B, T, head_size)
        return (out, wei) if return_weights else out


# ---------------------------------------------------------------- demo 1: the attention matrix
tokens = ["int", "x", "=", "5", ";", "return", "x", ";"]
T = len(tokens)
n_embd, head_size = 16, 8

embeddings = torch.randn(1, T, n_embd)   # random stand-ins (a real model LEARNS these)
head = Head(n_embd, head_size, block_size=T)
out, wei = head(embeddings, return_weights=True)

torch.set_printoptions(precision=2, linewidth=200, sci_mode=False)
print("Attention weights. Row = the token that is looking, column = the token it looks at.")
print("Each row sums to 1, and everything above the diagonal is 0 (the future is hidden):\n")
print("         " + "".join(f"{t:>8}" for t in tokens))
for i, t in enumerate(tokens):
    print(f"{t:>8} " + "".join(f"{w:8.2f}" for w in wei[0, i].tolist()))
print("\nrow sums:", wei[0].sum(dim=-1).tolist())
print("(untrained, so the weights are almost uniform. Training is what teaches the head that")
print(" the second 'return' row should put a big weight on 'x'.)")

# ---------------------------------------------------------------- demo 2: the future really is hidden
x2 = embeddings.clone()
x2[:, 5:] = torch.randn(1, T - 5, n_embd)          # change tokens 5, 6, 7
out2 = head(x2)
same_before = torch.allclose(out[:, :5], out2[:, :5], atol=1e-6)
print(f"\nchanged the LAST 3 tokens -> outputs of the first 5 tokens unchanged? {same_before}")

x3 = embeddings.clone()
x3[:, 1] = torch.randn(1, n_embd)                   # change token 1 ("x")
out3 = head(x3)
changed_after = not torch.allclose(out[:, 7], out3[:, 7], atol=1e-6)
print(f"changed token 1 -> output of the LAST token changed?               {changed_after}")

# ---------------------------------------------------------------- demo 3: why the scaling
q = torch.randn(1000, 64)
k = torch.randn(1000, 64)
raw = q @ k.T
scaled = raw * 64 ** -0.5
print("\nWhy we multiply by head_size ** -0.5 (head_size = 64):")
print(f"  score variance without scaling: {raw.var().item():6.1f}   (about 64)")
print(f"  score variance with scaling:    {scaled.var().item():6.1f}   (about 1)")
print(f"  biggest softmax weight in one row, no scaling: {F.softmax(raw[0], -1).max().item():.2f}")
print(f"  biggest softmax weight in one row, scaled:     {F.softmax(scaled[0], -1).max().item():.2f}")
print("  Without scaling the softmax collapses onto ONE token, and learning gets stuck.")

print("\nTry this:")
print("  1) Change head_size to 64 and look at the weight matrix again.")
print("  2) Delete the masked_fill line. Does the 'future hidden' test still pass? Why not?")
