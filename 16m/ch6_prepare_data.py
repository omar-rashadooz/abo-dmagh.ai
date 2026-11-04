"""Chapter 6 (practical): turn data/raw into training-ready token files.

What it does, in order:
  1. SCAN     walk data/raw/<category>/..., keep useful text/code files, drop junk
              (wrong extension, binary, too big/small, minified or auto-generated, exact duplicates)
  2. SPLIT    ~5% of the FILES go to validation (chosen by file hash, so no leakage between lines)
  3. TOKENIZE train a byte-level BPE tokenizer (5,000 tokens) on the TRAIN files only
  4. WRITE    data/clean/train.bin, val.bin (uint16 token ids), tokenizer.json, meta.json
  5. REPORT   how many tokens you really have, and how long training would take

Run from the project folder (the one that contains data/raw):
    python ch6_prepare_data.py

It reads about 1-2 GB of text and takes roughly 10-40 minutes. Safe to run again (it overwrites).
"""
import hashlib
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict, namedtuple
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "clean"

# ---------------------------------------------------------------- settings you may change
VOCAB_SIZE = 5000
EOT = "<|endoftext|>"                 # marks the end of every file
SEED = 1337
VAL_PERCENT = 5                       # share of FILES held out for validation
MIN_BYTES, MAX_BYTES = 200, 500_000   # skip tiny files and giant (usually generated) files
TOKENIZER_TRAIN_CHARS = 300_000_000   # the tokenizer learns from at most this much text
SUSTAINED_TPS_16M = 1261              # measured on your laptop (sustained_speed.py)

# how many times each source is repeated in train.bin (your priorities get more weight)
REPEAT = {"js": 3, "cpp": 3, "esp32": 2, "raspberry": 1, "c": 2, "python": 2,
          "hardware": 3, "me": 10}

PRESET_PARAMS = {"9M": 9_235_968, "16M": 16_205_568, "22M": 21_659_008, "28M": 27_898_880}

CODE_EXT = {".c", ".h", ".cpp", ".hpp", ".cc", ".cxx", ".hh", ".ino", ".py", ".js", ".mjs", ".cjs"}
SKIP_DIRS = {"node_modules", "third_party", "thirdparty", "vendor", "dist", "build",
             "__pycache__", "locale", "locales", "i18n", "po"}
SKIP_NAME_PREFIX = ("LICENSE", "COPYING", "NOTICE", "CHANGELOG", "CHANGES")

Doc = namedtuple("Doc", "path category sha split")


# ---------------------------------------------------------------- 1. scan
def accepted(path, category):
    name, ext = path.name, path.suffix.lower()
    if ".min." in name.lower() or name.upper().startswith(SKIP_NAME_PREFIX):
        return False
    if category in ("hardware", "me"):
        return ext in CODE_EXT or ext in {".txt", ".md"}
    return ext in CODE_EXT or ext == ".md" or name == "CMakeLists.txt"


def iter_files(category_dir):
    for dirpath, dirnames, filenames in os.walk(category_dir):
        dirnames[:] = sorted(d for d in dirnames
                             if d.lower() not in SKIP_DIRS and not d.startswith("."))
        for filename in sorted(filenames):
            yield Path(dirpath) / filename


def read_clean(path, category):
    """Return (clean_text, None) or (None, reason_it_was_skipped)."""
    try:
        size = path.stat().st_size
        if size < MIN_BYTES or size > MAX_BYTES:
            return None, "size"
        raw = path.read_bytes()
    except OSError:
        return None, "unreadable"
    if b"\x00" in raw:
        return None, "binary"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None, "not utf-8"

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{4,}", "\n\n\n", text).strip("\n") + "\n"
    if len(text) < MIN_BYTES:
        return None, "size"

    if category not in ("hardware", "me"):          # extra junk filters for repositories
        if not text.isascii():
            non_ascii = len(text) - len(text.encode("ascii", "ignore"))
            if non_ascii / len(text) > 0.2:
                return None, "mostly non-ascii"
        lines = text.split("\n")
        if max(map(len, lines)) > 3000:
            return None, "minified/generated"
        if len(lines) > 200:
            defines = sum(1 for line in lines if line.startswith("#define"))
            if defines / len(lines) > 0.7:
                return None, "minified/generated"   # register tables and the like
            if len(set(lines)) / len(lines) < 0.25:
                return None, "minified/generated"   # extremely repetitive
    return text, None


def collect(raw):
    docs, seen = [], set()
    skipped = Counter()
    stats = defaultdict(lambda: {"files": 0, "chars": 0})
    for category_dir in sorted(p for p in raw.iterdir() if p.is_dir()):
        category = category_dir.name
        for path in iter_files(category_dir):
            if not accepted(path, category):
                skipped["other extension"] += 1
                continue
            text, reason = read_clean(path, category)
            if reason:
                skipped[reason] += 1
                continue
            sha = hashlib.sha1(text.encode("utf-8")).hexdigest()
            if sha in seen:
                skipped["duplicate"] += 1
                continue
            seen.add(sha)
            split = "val" if int(sha[:8], 16) % 100 < VAL_PERCENT else "train"
            docs.append(Doc(path, category, sha, split))
            stats[category]["files"] += 1
            stats[category]["chars"] += len(text)
    return docs, skipped, stats


# ---------------------------------------------------------------- 3. tokenizer
def train_tokenizer(train_docs, path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    weighted = [d for d in train_docs for _ in range(REPEAT.get(d.category, 1))]
    random.Random(SEED).shuffle(weighted)

    def texts():
        total = 0
        for doc in weighted:
            text, _ = read_clean(doc.path, doc.category)
            yield text
            total += len(text)
            if total >= TOKENIZER_TRAIN_CHARS:
                break

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(texts(), trainer=trainer)
    tok.save(str(path))
    return tok


def encode_docs(tok, docs, eot_id, label):
    arrays = []
    start = time.time()
    for i in range(0, len(docs), 64):
        batch = docs[i:i + 64]
        texts = [read_clean(d.path, d.category)[0] for d in batch]
        for enc in tok.encode_batch(texts):
            arrays.append(np.array(enc.ids + [eot_id], dtype=np.uint16))
        if (i // 64) % 40 == 0:
            print(f"  {label}: {min(i + 64, len(docs)):,}/{len(docs):,} files "
                  f"({time.time() - start:.0f}s)", flush=True)
    return arrays


# ---------------------------------------------------------------- 4. write
def write_bin(path, arrays, docs, repeat, shuffle):
    order = [i for i, d in enumerate(docs) for _ in range(repeat.get(d.category, 1))]
    if shuffle:
        random.Random(SEED).shuffle(order)
    total = 0
    with open(path, "wb") as f:
        for i in order:
            f.write(arrays[i].tobytes())
            total += len(arrays[i])
    return total


# ---------------------------------------------------------------- main
def main():
    started = time.time()
    if not RAW.exists():
        sys.exit(f"{RAW} not found. Run this script from the project folder (the one with data/raw).")

    print("1/5 scanning data/raw ...")
    docs, skipped, stats = collect(RAW)
    train_docs = [d for d in docs if d.split == "train"]
    val_docs = [d for d in docs if d.split == "val"]
    print(f"kept {len(docs):,} files  ({len(train_docs):,} train / {len(val_docs):,} validation)")
    print("skipped:", ", ".join(f"{n:,} {why}" for why, n in skipped.most_common()) or "nothing")
    me_dir = RAW / "me"
    if not me_dir.exists() or not any(me_dir.rglob("*")):
        print("NOTE: data/raw/me is empty. Put your own notes (.txt/.md) there and run this again.")
    if not train_docs or not val_docs:
        sys.exit("not enough files for a train/validation split: check data/raw")

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"\n2/5 training the tokenizer ({VOCAB_SIZE:,} tokens) on the train files only ...")
    tok = train_tokenizer(train_docs, OUT / "tokenizer.json")
    eot_id = tok.token_to_id(EOT)

    print("\n3/5 turning files into token ids ...")
    train_arrays = encode_docs(tok, train_docs, eot_id, "train")
    val_arrays = encode_docs(tok, val_docs, eot_id, "val")

    print("\n4/5 writing data/clean/train.bin and val.bin ...")
    n_train = write_bin(OUT / "train.bin", train_arrays, train_docs, REPEAT, shuffle=True)
    n_val = write_bin(OUT / "val.bin", val_arrays, val_docs, {}, shuffle=False)

    print("\n5/5 report")
    unique = defaultdict(int)
    for d, arr in zip(train_docs, train_arrays):
        unique[d.category] += len(arr)
    print(f"{'source':<11}{'files':>8}{'MB':>9}{'unique tokens':>16}{'repeat':>8}{'effective':>14}")
    for category in sorted(stats):
        rep = REPEAT.get(category, 1)
        print(f"{category:<11}{stats[category]['files']:>8,}{stats[category]['chars'] / 1e6:>9.1f}"
              f"{unique[category]:>16,}{rep:>8}{unique[category] * rep:>14,}")
    unique_total = sum(unique.values())
    chars_total = sum(s["chars"] for s in stats.values())
    print(f"\nunique train tokens:    {unique_total:,}")
    print(f"effective train tokens: {n_train:,}   (train.bin, with the repeats)")
    print(f"validation tokens:      {n_val:,}")
    print(f"average: {chars_total / max(1, unique_total + n_val):.2f} characters per token")

    print("\nWhat size fits this data? (rule of thumb: ~20 tokens per parameter; time = ONE pass over train.bin)")
    print(f"{'preset':<8}{'params':>14}{'tokens wanted':>16}{'you have':>14}{'one pass':>12}")
    for name, params in PRESET_PARAMS.items():
        tps = SUSTAINED_TPS_16M * PRESET_PARAMS["16M"] / params
        hours = n_train / tps / 3600
        print(f"{name:<8}{params:>14,}{20 * params:>16,}{n_train:>14,}{hours / 6:>8.1f} nights")
    print("(speeds for 9M/22M/28M are scaled from your 16M measurement)")

    meta = {
        "vocab_size": VOCAB_SIZE, "eot_id": eot_id, "train_tokens": n_train,
        "unique_train_tokens": unique_total, "val_tokens": n_val,
        "repeat": REPEAT, "files": {"train": len(train_docs), "val": len(val_docs)},
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\ndone in {(time.time() - started) / 60:.1f} minutes. Files are in {OUT}")


if __name__ == "__main__":
    main()
