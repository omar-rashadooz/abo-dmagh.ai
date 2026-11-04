
import re
import json
import random
import numpy as np
from pathlib import Path

# ==========================================
# RASHADOOZ AI - CHARACTER DATA PREPARATION
# Version: Character-Level 1M
# ==========================================

SCRIPT_DIR = Path(__file__).resolve().parent

if (SCRIPT_DIR / "1.txt").exists():
    DATASET_DIR = SCRIPT_DIR
    BASE_DIR = SCRIPT_DIR.parent
else:
    BASE_DIR = SCRIPT_DIR
    DATASET_DIR = BASE_DIR / "Dataset"

DATA_DIR = BASE_DIR / "data_char"
DATA_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILES = [
    "1.txt",
    "2.txt",
    "3.txt",
    "4.txt",
    "5.txt"
]

TRAIN_RATIO = 0.90
RANDOM_SEED = 42
TARGET_PARAMETERS = 1_000_000

# Model architecture
BLOCK_SIZE = 256
N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.1

UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [
    UNK_TOKEN,
    PAD_TOKEN,
    BOS_TOKEN,
    EOS_TOKEN
]

TRAIN_FILE = DATA_DIR / "train.bin"
VAL_FILE = DATA_DIR / "val.bin"
VOCAB_FILE = DATA_DIR / "vocab.json"
CONFIG_FILE = DATA_DIR / "config.json"


# ==========================================
# 1. READ DATA
# ==========================================

def read_files():
    documents = []

    for filename in INPUT_FILES:
        path = DATASET_DIR / filename

        if not path.exists():
            print(f"[WARNING] Missing file: {filename}")
            continue

        text = path.read_text(encoding="utf-8")

        if text.strip():
            documents.append((filename, text))
            print(
                f"[READ] {filename}: "
                f"{len(text):,} characters"
            )
        else:
            print(f"[WARNING] Empty file: {filename}")

    if not documents:
        raise RuntimeError("No training text files found.")

    return documents


# ==========================================
# 2. SPLIT INTO SAMPLES
# ==========================================

def split_samples(documents):
    grouped = {}

    for filename, document in documents:

        parts = re.split(r"\n\s*\n", document)

        samples = [
            part.strip()
            for part in parts
            if part.strip()
        ]

        grouped[filename] = samples

        print(
            f"[INFO] {filename}: "
            f"{len(samples)} samples"
        )

    train_samples = []
    val_samples = []

    rng = random.Random(RANDOM_SEED)

    for filename, samples in grouped.items():

        samples = samples.copy()
        rng.shuffle(samples)

        if len(samples) < 2:
            train_samples.extend(samples)
            continue

        val_count = max(
            1,
            int(round(len(samples) * (1 - TRAIN_RATIO)))
        )

        val_count = min(val_count, len(samples) - 1)

        val_samples.extend(samples[:val_count])
        train_samples.extend(samples[val_count:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)

    if not train_samples or not val_samples:
        raise RuntimeError("Training or validation set is empty.")

    print(f"\n[INFO] Training samples: {len(train_samples)}")
    print(f"[INFO] Validation samples: {len(val_samples)}")

    return train_samples, val_samples


# ==========================================
# 3. CHARACTER TOKENIZER
# ==========================================

def tokenize(text):
    return list(text)


# ==========================================
# 4. BUILD CHARACTER VOCABULARY
# ==========================================

def build_vocab(all_samples):

    characters = set()

    for sample in all_samples:
        characters.update(tokenize(sample))

    vocab = {}

    for token in SPECIAL_TOKENS:
        vocab[token] = len(vocab)

    for character in sorted(characters):
        if character not in vocab:
            vocab[character] = len(vocab)

    print(f"[INFO] Character vocabulary: {len(vocab)}")

    return vocab


# ==========================================
# 5. MODEL CONFIGURATION
# ==========================================

def calculate_ffn_size(vocab_size):

    # Exact parameter formula for this architecture:
    # Parameters = 299776 + 1028*FFN_SIZE + 129*VOCAB_SIZE

    fixed_parameters = 299776

    ffn_size = round(
        (
            TARGET_PARAMETERS
            - fixed_parameters
            - 129 * vocab_size
        ) / 1028
    )

    return max(1, ffn_size)


def calculate_parameters(vocab_size, ffn_size):

    return (
        299776
        + 1028 * ffn_size
        + 129 * vocab_size
    )


# ==========================================
# 6. ENCODE SAMPLES
# ==========================================

def encode_sample(sample, vocab):

    ids = [vocab[BOS_TOKEN]]

    for character in tokenize(sample):
        ids.append(
            vocab.get(character, vocab[UNK_TOKEN])
        )

    ids.append(vocab[EOS_TOKEN])

    return ids


# ==========================================
# 7. SAVE BINARY DATA
# ==========================================

def save_binary(samples, vocab, output_file):

    all_ids = []

    for sample in samples:
        all_ids.extend(
            encode_sample(sample, vocab)
        )

    data = np.asarray(all_ids, dtype=np.uint16)

    data.tofile(output_file)

    print(
        f"[SAVED] {output_file.name}: "
        f"{len(data):,} characters/tokens | "
        f"{output_file.stat().st_size / 1024:.2f} KB"
    )


# ==========================================
# 8. MAIN
# ==========================================

def main():

    print("\n======================================")
    print(" RASHADOOZ AI - CHARACTER PREPARATION")
    print("======================================\n")

    documents = read_files()

    train_samples, val_samples = split_samples(documents)

    all_samples = train_samples + val_samples

    # Build vocabulary from all characters.
    # Character vocabulary does not expose word-level
    # semantic information from validation.
    vocab = build_vocab(all_samples)

    vocab_size = len(vocab)

    ffn_size = calculate_ffn_size(vocab_size)

    actual_parameters = calculate_parameters(
        vocab_size,
        ffn_size
    )

    config = {
        "vocab_size": vocab_size,
        "block_size": BLOCK_SIZE,
        "n_embd": N_EMBD,
        "n_head": N_HEAD,
        "n_layer": N_LAYER,
        "ffn_size": ffn_size,
        "dropout": DROPOUT,
        "target_parameters": TARGET_PARAMETERS,
        "actual_parameters": actual_parameters,
        "tokenizer": "character-level"
    }

    # Save vocabulary
    with open(VOCAB_FILE, "w", encoding="utf-8") as f:
        json.dump(
            vocab,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"[SAVED] {VOCAB_FILE.name}")

    # Save model configuration
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(
            config,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(f"[SAVED] {CONFIG_FILE.name}")

    # Save datasets
    save_binary(train_samples, vocab, TRAIN_FILE)
    save_binary(val_samples, vocab, VAL_FILE)

    print("\n======================================")
    print(" CHARACTER DATA PREPARATION COMPLETE")
    print("======================================")

    print("Vocabulary:", vocab_size)
    print("FFN size:", ffn_size)
    print("Target parameters:", TARGET_PARAMETERS)
    print("Actual parameters:", f"{actual_parameters:,}")

    print("\nGenerated:")
    print(" - data_char/train.bin")
    print(" - data_char/val.bin")
    print(" - data_char/vocab.json")
    print(" - data_char/config.json")


if __name__ == "__main__":
    main()
