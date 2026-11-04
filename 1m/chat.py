
import json
import re
import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path


# ==========================================
# RASHADOOZ AI - CHARACTER LEVEL CHAT
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data_char"
OUT_DIR = BASE_DIR / "out_char"

VOCAB_PATH = DATA_DIR / "vocab.json"
CONFIG_PATH = DATA_DIR / "config.json"

BEST_CHECKPOINT = OUT_DIR / "best.pt"
FINAL_CHECKPOINT = OUT_DIR / "final.pt"
LATEST_CHECKPOINT = OUT_DIR / "last.pt"

DEVICE = torch.device("cpu")

torch.set_num_threads(2)


# ==========================================
# 1. LOAD CONFIGURATION
# ==========================================

if not CONFIG_PATH.exists():
    raise FileNotFoundError(
        "data_char/config.json not found."
    )

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
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
# 2. LOAD VOCABULARY
# ==========================================

if not VOCAB_PATH.exists():
    raise FileNotFoundError(
        "data_char/vocab.json not found."
    )

with open(VOCAB_PATH, "r", encoding="utf-8") as f:
    vocab = json.load(f)

if len(vocab) != VOCAB_SIZE:
    raise ValueError("Vocabulary size mismatch.")

stoi = vocab

itos = {
    int(index): token
    for token, index in vocab.items()
}

UNK_ID = stoi["<unk>"]
BOS_ID = stoi["<bos>"]
EOS_ID = stoi["<eos>"]


# ==========================================
# 3. CHARACTER TOKENIZER
# ==========================================

def tokenize(text):
    return list(text)


def encode(text):

    return [
        stoi.get(character, UNK_ID)
        for character in tokenize(text)
    ]


def decode(ids):

    characters = []

    for token_id in ids:

        token = itos.get(int(token_id), "<unk>")

        if token in ("<bos>", "<eos>", "<pad>"):
            continue

        characters.append(token)

    return "".join(characters)


# ==========================================
# 4. CAUSAL SELF ATTENTION
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
# 5. FEED FORWARD
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
# 6. TRANSFORMER BLOCK
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
# 7. RASHADOOZ GPT
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
# 8. LOAD CHECKPOINT
# ==========================================

if BEST_CHECKPOINT.exists():
    CHECKPOINT_PATH = BEST_CHECKPOINT

elif FINAL_CHECKPOINT.exists():
    CHECKPOINT_PATH = FINAL_CHECKPOINT

elif LATEST_CHECKPOINT.exists():
    CHECKPOINT_PATH = LATEST_CHECKPOINT

else:
    raise FileNotFoundError(
        "No Character-Level checkpoint found."
    )

print("[LOAD] Loading:", CHECKPOINT_PATH.name)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

if checkpoint.get("config") != MODEL_CONFIG:
    raise ValueError("Checkpoint configuration mismatch.")

model = RashadoozGPT().to(DEVICE)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

parameter_count = sum(
    p.numel() for p in model.parameters()
)

print("[OK] Model loaded.")
print("[OK] Tokenizer: Character-Level")
print("[OK] Parameters:", f"{parameter_count:,}")
print("[OK] Training step:", checkpoint.get("step", "unknown"))


# ==========================================
# 9. TEXT GENERATION
# ==========================================

@torch.no_grad()
def generate(
    prompt,
    max_new_tokens=120,
    temperature=0.7,
    top_k=20
):

    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")

    if max_new_tokens < 1:
        return ""

    # Training samples use:
    # BOS + question + newline + answer + EOS

    if not prompt.endswith("\n"):
        prompt += "\n"

    prompt_ids = [BOS_ID] + encode(prompt)

    idx = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=DEVICE
    )

    generated = []

    for _ in range(max_new_tokens):

        idx_cond = idx[:, -BLOCK_SIZE:]

        logits = model(idx_cond)

        logits = logits[:, -1, :]

        logits = logits / temperature

        if top_k is not None:

            k = min(top_k, logits.size(-1))

            values, _ = torch.topk(logits, k)

            threshold = values[:, [-1]]

            logits = logits.masked_fill(
                logits < threshold,
                float("-inf")
            )

        probs = F.softmax(logits, dim=-1)

        next_token = torch.multinomial(
            probs,
            num_samples=1
        )

        token_id = next_token.item()

        if token_id == EOS_ID:
            break

        generated.append(token_id)

        idx = torch.cat(
            (idx, next_token),
            dim=1
        )

    return decode(generated).strip()


# ==========================================
# 10. CHAT INTERFACE
# ==========================================

def main():

    print("\n======================================")
    print(" RASHADOOZ AI - CHARACTER LEVEL CHAT")
    print("======================================")

    print("Type 'exit' or 'quit' to close.")
    print("Type '/help' for commands.")

    temperature = 0.7
    top_k = 20
    max_tokens = 120

    while True:

        try:

            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            command = user_input.lower()

            if command in ["exit", "quit"]:
                print("\nRashadooz AI: Goodbye!")
                break

            if command == "/help":

                print("\nCommands:")
                print("/help       - Show commands")
                print("/temp VALUE - Set temperature")
                print("/topk VALUE - Set top-k")
                print("/tokens N   - Set response length")
                print("exit        - Exit chat")

                continue

            if command.startswith("/temp "):

                try:

                    temperature = float(user_input.split()[1])

                    if temperature <= 0:
                        raise ValueError

                    print("Temperature:", temperature)

                except (ValueError, IndexError):

                    print("Invalid temperature.")

                continue

            if command.startswith("/topk "):

                try:

                    top_k = int(user_input.split()[1])

                    if top_k < 1:
                        raise ValueError

                    print("Top-k:", top_k)

                except (ValueError, IndexError):

                    print("Invalid top-k.")

                continue

            if command.startswith("/tokens "):

                try:

                    max_tokens = int(user_input.split()[1])

                    if not 1 <= max_tokens <= 256:
                        raise ValueError

                    print("Maximum tokens:", max_tokens)

                except (ValueError, IndexError):

                    print("Token count must be between 1 and 256.")

                continue

            print("\nRashadooz AI: ", end="", flush=True)

            response = generate(
                user_input,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k
            )

            print(
                response
                if response
                else "[No response generated]"
            )

        except KeyboardInterrupt:

            print("\n\n[CHAT CLOSED]")
            break

        except EOFError:

            print("\n\n[CHAT CLOSED]")
            break


if __name__ == "__main__":
    main()
