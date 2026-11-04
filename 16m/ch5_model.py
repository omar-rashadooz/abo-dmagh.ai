"""Chapter 5 (practical): the full GPT-style model, built from scratch.

    token ids -> token embedding + position embedding
              -> N x [ LayerNorm -> multi-head causal attention -> (+ residual)
                       LayerNorm -> feed-forward MLP             -> (+ residual) ]
              -> LayerNorm -> output layer -> a score for every token in the vocabulary

Run:
    python ch5_model.py          # the 16M preset  (n_embd 384, 8 layers, 6 heads)
    python ch5_model.py 9M       # the 9M preset   (n_embd 256, 10 layers, 8 heads)

It prints: the parameter count, a few sanity checks, and how fast YOUR cpu trains this model
(tokens per second), then converts that into hours / nights for 100M and 200M training tokens.
The speed test takes a couple of minutes: be patient.
"""
import math
import sys
import time
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(1337)


@dataclass
class Config:
    vocab_size: int = 5000     # decided in chapter 2
    block_size: int = 256      # context length: how many tokens the model sees at once
    n_layer: int = 8           # number of transformer blocks
    n_head: int = 6            # attention heads per block
    n_embd: int = 384          # size of every token vector
    fast_attention: bool = False   # True = PyTorch's built-in attention kernel


PRESETS = {
    "28M": Config(n_embd=512, n_layer=8, n_head=8),
    "22M": Config(n_embd=448, n_layer=8, n_head=7),
    "16M": Config(n_embd=384, n_layer=8, n_head=6),
    "9M": Config(n_embd=256, n_layer=10, n_head=8),
}


class CausalSelfAttention(nn.Module):
    """Chapter 4's attention, but with several heads computed at once."""

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0, "n_embd must be divisible by n_head"
        self.n_head = cfg.n_head
        self.fast = cfg.fast_attention
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)  # Query, Key, Value together
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)                  # mixes the heads' outputs
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        self.register_buffer("tril", mask.view(1, 1, cfg.block_size, cfg.block_size))

    def forward(self, x):
        B, T, C = x.shape
        hs = C // self.n_head                                  # size of each head
        q, k, v = self.qkv(x).split(C, dim=2)                  # three tensors of (B, T, C)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)      # (B, heads, T, hs)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        if self.fast:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            att = q @ k.transpose(-2, -1) * hs ** -0.5         # (B, heads, T, T)
            att = att.masked_fill(self.tril[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            y = att @ v                                         # (B, heads, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)        # glue the heads back together
        return self.proj(y)


class MLP(nn.Module):
    """The per-token 'thinking' layer: widen x4, apply GELU, shrink back."""

    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))   # residual connection: keep the old info, ADD the new
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)   # which token is it?
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)   # where is it?
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight                    # weight tying: saves V x d numbers
        self.apply(self._init_weights)
        for name, p in self.named_parameters():                   # calmer start for residual outputs
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, "sequence longer than block_size"
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)                 # (B, T, C)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)                                     # (B, T, vocab)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size:])
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


def expected_params(cfg):
    """Pencil-and-paper count. Must match the real count."""
    d = cfg.n_embd
    per_block = 12 * d * d + 10 * d      # attention 4d^2+d, MLP 8d^2+5d, two LayerNorms 4d
    return cfg.n_layer * per_block + cfg.vocab_size * d + cfg.block_size * d + 2 * d


def benchmark(cfg, batch=8, steps=3):
    """Real training steps on random tokens: returns tokens processed per second."""
    model = GPT(cfg)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randint(0, cfg.vocab_size, (batch, cfg.block_size))
    y = torch.randint(0, cfg.vocab_size, (batch, cfg.block_size))

    def one_step():
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    one_step()                           # warm-up (not timed)
    t0 = time.time()
    for _ in range(steps):
        one_step()
    return batch * cfg.block_size * steps / (time.time() - t0)


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "16M"
    cfg = PRESETS[name]
    model = GPT(cfg)

    # ---------------------------------------------------------------- 1. size
    total = sum(p.numel() for p in model.parameters())   # tied weights are counted once
    emb = model.tok_emb.weight.numel() + model.pos_emb.weight.numel()
    blocks = sum(p.numel() for p in model.blocks.parameters())
    print(f"preset {name}: {cfg}")
    print(f"parameters: {total:,}")
    print(f"  embeddings (token + position): {emb:,}")
    print(f"  {cfg.n_layer} transformer blocks:          {blocks:,}")
    print(f"  pencil-and-paper formula says: {expected_params(cfg):,}   "
          f"-> match: {total == expected_params(cfg)}")

    # ---------------------------------------------------------------- 2. sanity checks
    idx = torch.randint(0, cfg.vocab_size, (4, 64))
    tgt = torch.randint(0, cfg.vocab_size, (4, 64))
    _, loss = model(idx, tgt)
    print(f"\ninitial loss on random data: {loss.item():.3f}   (random guess = ln({cfg.vocab_size}) = "
          f"{math.log(cfg.vocab_size):.3f})")

    model.eval()
    a = torch.randint(0, cfg.vocab_size, (1, 40))
    b = a.clone()
    b[:, 25:] = torch.randint(0, cfg.vocab_size, (1, 15))
    with torch.no_grad():
        la, _ = model(a)
        lb, _ = model(b)
    print(f"changing the last 15 tokens leaves the first 25 predictions unchanged? "
          f"{torch.allclose(la[:, :25], lb[:, :25], atol=1e-4)}")

    out = model.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=10)
    print(f"generate() works, 10 new token ids (untrained = random): {out[0, 1:].tolist()}")

    # ---------------------------------------------------------------- 3. speed of YOUR cpu
    print(f"\nspeed test on {torch.get_num_threads()} cpu threads (a couple of minutes)...")
    speeds = {}
    for label, fast in (("manual attention", False), ("built-in attention", True)):
        speeds[label] = benchmark(replace(cfg, fast_attention=fast))
        print(f"  {label:18s}: {speeds[label]:7.0f} tokens/second")
    best_label = max(speeds, key=speeds.get)
    tps = speeds[best_label]
    print(f"\nfastest: {best_label}")
    print("estimated training time with the fastest one (6 hours per night):")
    for tokens in (100e6, 200e6):
        hours = tokens / tps / 3600
        print(f"  {tokens / 1e6:.0f}M tokens -> {hours:6.0f} hours = {hours / 6:5.1f} nights")
    print("\n(real training adds a little overhead for data loading and checkpoints)")
