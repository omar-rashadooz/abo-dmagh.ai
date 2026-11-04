
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path


# ==========================================
# RASHADOOZ AI - CHARACTER LEVEL TRAINING
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data_char"
OUT_DIR = BASE_DIR / "out_char"

OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = DATA_DIR / "train.bin"
VAL_FILE = DATA_DIR / "val.bin"
CONFIG_FILE = DATA_DIR / "config.json"

LAST_CHECKPOINT = OUT_DIR / "last.pt"
BEST_CHECKPOINT = OUT_DIR / "best.pt"
FINAL_CHECKPOINT = OUT_DIR / "final.pt"

DEVICE = torch.device("cpu")

torch.set_num_threads(4)

# ==========================================
# 1. TRAINING SETTINGS
# ==========================================

BATCH_SIZE = 16
MAX_ITERS = 6000

EVAL_INTERVAL = 200
EVAL_ITERS = 30

CHECKPOINT_INTERVAL = 100
LOG_INTERVAL = 10

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

# ==========================================
# 2. LOAD CONFIGURATION
# ==========================================

if not CONFIG_FILE.exists():
    raise FileNotFoundError(
        "data_char/config.json not found. Run prepare.py first."
    )

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = json.load(f)

VOCAB_SIZE = config["vocab_size"]
BLOCK_SIZE = config["block_size"]
N_EMBD = config["n_embd"]
N_HEAD = config["n_head"]
N_LAYER = config["n_layer"]
FFN_SIZE = config["ffn_size"]
DROPOUT = config["dropout"]

MODEL_CONFIG = {
    "vocab_size": VOCAB_SIZE,
    "block_size": BLOCK_SIZE,
    "n_embd": N_EMBD,
    "n_head": N_HEAD,
    "n_layer": N_LAYER,
    "ffn_size": FFN_SIZE,
    "dropout": DROPOUT,
    "tokenizer": "character-level"
}

# ==========================================
# 3. LOAD DATA
# ==========================================

if not TRAIN_FILE.exists() or not VAL_FILE.exists():
    raise FileNotFoundError(
        "Training data missing. Run prepare.py first."
    )

train_np = np.fromfile(TRAIN_FILE, dtype=np.uint16)
val_np = np.fromfile(VAL_FILE, dtype=np.uint16)

train_data = torch.from_numpy(
    train_np.astype(np.int64)
)

val_data = torch.from_numpy(
    val_np.astype(np.int64)
)

if len(train_data) <= BLOCK_SIZE + 1:
    raise RuntimeError("Training data is too small.")

if len(val_data) <= BLOCK_SIZE + 1:
    raise RuntimeError("Validation data is too small.")

print("[DATA] Train tokens:", len(train_data))
print("[DATA] Val tokens:", len(val_data))
print("[DATA] Vocabulary:", VOCAB_SIZE)


# ==========================================
# 4. GET BATCH
# ==========================================

def get_batch(split):

    data = train_data if split == "train" else val_data

    ix = torch.randint(
        len(data) - BLOCK_SIZE - 1,
        (BATCH_SIZE,)
    )

    x = torch.stack([
        data[int(i):int(i) + BLOCK_SIZE]
        for i in ix
    ])

    y = torch.stack([
        data[int(i) + 1:int(i) + BLOCK_SIZE + 1]
        for i in ix
    ])

    return x.to(DEVICE), y.to(DEVICE)


# ==========================================
# 5. CAUSAL SELF ATTENTION
# ==========================================

class CausalSelfAttention(nn.Module):

    def __init__(self):

        super().__init__()

        self.head_size = N_EMBD // N_HEAD

        self.qkv = nn.Linear(
            N_EMBD,
            3 * N_EMBD
        )

        self.proj = nn.Linear(
            N_EMBD,
            N_EMBD
        )

        self.attn_dropout = nn.Dropout(DROPOUT)
        self.resid_dropout = nn.Dropout(DROPOUT)

        self.register_buffer(
            "mask",
            torch.tril(
                torch.ones(BLOCK_SIZE, BLOCK_SIZE)
            ).view(1, 1, BLOCK_SIZE, BLOCK_SIZE)
        )

    def forward(self, x):

        B, T, C = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.split(N_EMBD, dim=2)

        q = q.view(
            B, T, N_HEAD, self.head_size
        ).transpose(1, 2)

        k = k.view(
            B, T, N_HEAD, self.head_size
        ).transpose(1, 2)

        v = v.view(
            B, T, N_HEAD, self.head_size
        ).transpose(1, 2)

        scores = (
            q @ k.transpose(-2, -1)
        ) * (self.head_size ** -0.5)

        scores = scores.masked_fill(
            self.mask[:, :, :T, :T] == 0,
            float("-inf")
        )

        weights = F.softmax(scores, dim=-1)

        weights = self.attn_dropout(weights)

        output = weights @ v

        output = output.transpose(
            1, 2
        ).contiguous().view(B, T, C)

        output = self.proj(output)

        return self.resid_dropout(output)


# ==========================================
# 6. FEED FORWARD
# ==========================================

class FeedForward(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(N_EMBD, FFN_SIZE),
            nn.GELU(),
            nn.Linear(FFN_SIZE, N_EMBD),
            nn.Dropout(DROPOUT)
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# 7. TRANSFORMER BLOCK
# ==========================================

class TransformerBlock(nn.Module):

    def __init__(self):

        super().__init__()

        self.ln1 = nn.LayerNorm(N_EMBD)
        self.attention = CausalSelfAttention()

        self.ln2 = nn.LayerNorm(N_EMBD)
        self.feed_forward = FeedForward()

    def forward(self, x):

        x = x + self.attention(self.ln1(x))

        x = x + self.feed_forward(self.ln2(x))

        return x


# ==========================================
# 8. RASHADOOZ GPT
# ==========================================

class RashadoozGPT(nn.Module):

    def __init__(self):

        super().__init__()

        self.token_embedding = nn.Embedding(
            VOCAB_SIZE,
            N_EMBD
        )

        self.position_embedding = nn.Embedding(
            BLOCK_SIZE,
            N_EMBD
        )

        self.dropout = nn.Dropout(DROPOUT)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock()
                for _ in range(N_LAYER)
            ]
        )

        self.final_ln = nn.LayerNorm(N_EMBD)

        self.lm_head = nn.Linear(
            N_EMBD,
            VOCAB_SIZE
        )

        self.lm_head.weight = self.token_embedding.weight

    def forward(self, idx):

        B, T = idx.shape

        positions = torch.arange(
            T,
            device=idx.device
        )

        tok_emb = self.token_embedding(idx)

        pos_emb = self.position_embedding(positions)

        x = self.dropout(tok_emb + pos_emb)

        x = self.blocks(x)

        x = self.final_ln(x)

        return self.lm_head(x)


# ==========================================
# 9. INITIALIZE MODEL
# ==========================================

model = RashadoozGPT().to(DEVICE)

parameter_count = sum(
    p.numel() for p in model.parameters()
)

print("[MODEL] Parameters:", f"{parameter_count:,}")
print("[MODEL] Device:", DEVICE)


# ==========================================
# 10. OPTIMIZER
# ==========================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ==========================================
# 11. EVALUATION
# ==========================================

@torch.no_grad()
def estimate_loss():

    model.eval()

    results = {}

    for split in ["train", "val"]:

        losses = torch.zeros(EVAL_ITERS)

        for k in range(EVAL_ITERS):

            X, Y = get_batch(split)

            logits = model(X)

            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                Y.reshape(-1)
            )

            losses[k] = loss.item()

        results[split] = losses.mean().item()

    model.train()

    return results


# ==========================================
# 12. CHECKPOINT FUNCTIONS
# ==========================================

def save_checkpoint(path, step, best_val_loss):

    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "config": MODEL_CONFIG
    }

    torch.save(checkpoint, path)

    print(
        f"[CHECKPOINT] Saved: {path.name} | Step {step}"
    )


# ==========================================
# 13. RESUME
# ==========================================

start_step = 0
best_val_loss = float("inf")

if LAST_CHECKPOINT.exists():

    print("[RESUME] Loading:", LAST_CHECKPOINT.name)

    checkpoint = torch.load(
        LAST_CHECKPOINT,
        map_location=DEVICE,
        weights_only=False
    )

    if checkpoint.get("config") != MODEL_CONFIG:
        raise ValueError(
            "Checkpoint configuration mismatch. "
            "Do not mix different datasets or tokenizers."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    start_step = checkpoint["step"]
    best_val_loss = checkpoint["best_val_loss"]

    print("[RESUME] Step:", start_step)

else:
    print("[INFO] Starting new Character-Level training.")


# ==========================================
# 14. TRAINING LOOP
# ==========================================

def main():

    global best_val_loss

    session_start = time.time()
    session_steps = 0

    model.train()

    print("\n======================================")
    print(" RASHADOOZ CHARACTER-LEVEL TRAINING")
    print("======================================")

    print("Start step:", start_step)
    print("Target steps:", MAX_ITERS)

    try:

        for step in range(start_step, MAX_ITERS):

            current_step = step + 1

            # Evaluation
            if current_step % EVAL_INTERVAL == 0:

                losses = estimate_loss()

                print(
                    f"\nStep {current_step}/{MAX_ITERS} | "
                    f"Train: {losses['train']:.4f} | "
                    f"Val: {losses['val']:.4f}"
                )

                if losses["val"] < best_val_loss:

                    best_val_loss = losses["val"]

                    save_checkpoint(
                        BEST_CHECKPOINT,
                        current_step,
                        best_val_loss
                    )

            # Training batch
            X, Y = get_batch("train")

            logits = model(X)

            loss = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                Y.reshape(-1)
            )

            optimizer.zero_grad(set_to_none=True)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP
            )

            optimizer.step()

            session_steps += 1

            if current_step % LOG_INTERVAL == 0:

                elapsed = time.time() - session_start

                speed = (
                    session_steps / elapsed
                    if elapsed > 0 else 0
                )

                remaining = MAX_ITERS - current_step

                eta = (
                    remaining / speed
                    if speed > 0 else 0
                )

                print(
                    f"Step {current_step}/{MAX_ITERS} | "
                    f"Loss: {loss.item():.4f}"
                )

                print(
                    f"[SPEED] {speed:.2f} steps/sec | "
                    f"Progress: {current_step / MAX_ITERS * 100:.2f}% | "
                    f"ETA: {eta / 60:.1f} min"
                )

            if current_step % CHECKPOINT_INTERVAL == 0:

                save_checkpoint(
                    LAST_CHECKPOINT,
                    current_step,
                    best_val_loss
                )

        # Final evaluation
        losses = estimate_loss()

        print("\n======================================")
        print(" FINAL EVALUATION")
        print("======================================")

        print("Train Loss:", round(losses["train"], 4))
        print("Val Loss:", round(losses["val"], 4))

        if losses["val"] < best_val_loss:

            best_val_loss = losses["val"]

            save_checkpoint(
                BEST_CHECKPOINT,
                MAX_ITERS,
                best_val_loss
            )

        save_checkpoint(
            LAST_CHECKPOINT,
            MAX_ITERS,
            best_val_loss
        )

        save_checkpoint(
            FINAL_CHECKPOINT,
            MAX_ITERS,
            best_val_loss
        )

        elapsed = time.time() - session_start

        print("\n======================================")
        print("       TRAINING COMPLETE")
        print("======================================")

        print("Current step:", MAX_ITERS)
        print("Best validation loss:", best_val_loss)
        print("Session steps:", session_steps)
        print("Session time:", f"{elapsed / 60:.2f} minutes")
        print("Checkpoint:", LAST_CHECKPOINT)

    except KeyboardInterrupt:

        print("\n[INTERRUPTED] Saving training state.")

        save_checkpoint(
            LAST_CHECKPOINT,
            current_step - 1,
            best_val_loss
        )

        print("[INFO] You can resume using: python train.py")


if __name__ == "__main__":
    main()
