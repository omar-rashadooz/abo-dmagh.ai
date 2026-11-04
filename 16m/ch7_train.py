"""Chapter 7: train abo-dmagh.ai.

    python ch7_train.py                      # 16M model, resumes automatically if a checkpoint exists
    python ch7_train.py --stop-at 02:00      # stop (and save) when the clock reaches 02:00
    python ch7_train.py --preset 9M --max-steps 250 --minutes 25 --eval-every 100 --out runs/pilot --fresh
                                             # a short PILOT run to check that everything works

Needs the files made by ch6_prepare_data.py (data/clean/*).
Everything is saved in runs/<name>/ :
    ckpt_last.pt   full state (model + optimizer + step): used to resume. ckpt_prev.pt = the one before.
    model_best.pt  weights with the best validation loss so far
    log.csv        step, tokens, train loss, validation loss, learning rate, time
Press Ctrl+C once to stop cleanly (it saves a checkpoint first).
"""
import argparse
import json
import math
import os
import signal
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ch5_model import GPT, PRESETS
from train_utils import deadline, hours_text, lr_at, sample_windows, warmup_steps

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "clean"

PROMPTS = [
    "#include <stdio.h>\n\nint main(",
    "#include <WiFi.h>\n\nvoid setup() {\n  Serial.begin(115200);\n",
    "const express = require(",
    "#include <iostream>\n\nint main() {\n",
    "import machine\nfrom machine import Pin\n",
]


def parse_args():
    p = argparse.ArgumentParser(description="train abo-dmagh.ai")
    p.add_argument("--preset", default="16M", choices=sorted(PRESETS))
    p.add_argument("--out", default=None, help="run folder (default runs/abo-dmagh-<preset>)")
    p.add_argument("--stop-at", default=None, help="clock time to stop at, e.g. 02:00")
    p.add_argument("--minutes", type=float, default=None, help="stop after this many minutes")
    p.add_argument("--max-steps", type=int, default=None, help="default: one pass over train.bin")
    p.add_argument("--batch", type=int, default=8, help="windows per micro-batch")
    p.add_argument("--accum", type=int, default=4, help="micro-batches per optimizer step")
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--min-lr", type=float, default=6e-5)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-iters", type=int, default=20)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--ckpt-minutes", type=float, default=20)
    p.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint and start over")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def atomic_save(obj, path):
    """Write to a temporary file first, so a power cut can never leave a half-written checkpoint."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(out, model, opt, step, tokens_seen, best_val, cfg, tokens_per_step):
    state = {
        "model": model.state_dict(), "optimizer": opt.state_dict(),
        "step": step, "tokens_seen": tokens_seen, "best_val": best_val,
        "config": asdict(cfg), "tokens_per_step": tokens_per_step,
    }
    tmp = out / "ckpt_last.pt.tmp"
    torch.save(state, tmp)
    last, prev = out / "ckpt_last.pt", out / "ckpt_prev.pt"
    if last.exists():
        os.replace(last, prev)
    os.replace(tmp, last)


def sample_text(model, tok, cfg, prompt, new_tokens=120, temperature=0.8, top_k=50):
    ids = tok.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        for _ in range(new_tokens):
            logits, _ = model(idx[:, -cfg.block_size:])
            logits = logits[:, -1, :] / temperature
            kth = torch.topk(logits, top_k).values[:, -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
    model.train()
    return tok.decode(idx[0].tolist())


def main():
    args = parse_args()
    cfg = replace(PRESETS[args.preset], fast_attention=True)
    out = Path(args.out) if args.out else ROOT / "runs" / f"abo-dmagh-{args.preset}"
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ data
    for name in ("train.bin", "val.bin", "meta.json", "tokenizer.json"):
        if not (DATA / name).exists():
            sys.exit(f"{DATA / name} is missing: run ch6_prepare_data.py first")
    meta = json.loads((DATA / "meta.json").read_text(encoding="utf-8"))
    if meta["vocab_size"] != cfg.vocab_size:
        sys.exit(f"tokenizer has {meta['vocab_size']} tokens but the model expects {cfg.vocab_size}")
    train_data = np.memmap(DATA / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(DATA / "val.bin", dtype=np.uint16, mode="r")
    T = cfg.block_size
    if len(train_data) <= T + 2 or len(val_data) <= T + 2:
        sys.exit("train.bin or val.bin is too small: check the output of ch6_prepare_data.py")

    tok = None
    try:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(DATA / "tokenizer.json"))
    except Exception as error:   # samples are a bonus: training must not depend on them
        print(f"(text samples disabled: {error})")

    tokens_per_step = args.batch * args.accum * T
    max_steps = args.max_steps or len(train_data) // tokens_per_step
    warmup = warmup_steps(max_steps)

    # ------------------------------------------------------------ model + optimizer
    model = GPT(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    decay = [p for p in model.parameters() if p.dim() >= 2]       # weight matrices
    no_decay = [p for p in model.parameters() if p.dim() < 2]     # biases and LayerNorm
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )

    step, tokens_seen, best_val = 0, 0, float("inf")
    last = out / "ckpt_last.pt"
    if last.exists() and not args.fresh:
        ck = torch.load(last, map_location="cpu")
        if ck["config"] != asdict(cfg):
            sys.exit("the checkpoint was made with a different model size: use another --out or --fresh")
        if ck["tokens_per_step"] != tokens_per_step:
            sys.exit("batch/accum changed since the checkpoint: use the same values, or --fresh")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        step, tokens_seen, best_val = ck["step"], ck["tokens_seen"], ck["best_val"]
        print(f"resumed from {last} at step {step:,} ({tokens_seen / 1e6:.1f}M tokens seen)")
    elif last.exists():
        print("--fresh: starting over (the old checkpoint will be replaced at the first save)")

    # ------------------------------------------------------------ stop conditions
    stop = {"flag": False}

    def request_stop(signum, frame):
        if stop["flag"]:
            sys.exit(1)
        stop["flag"] = True
        print("\nstop requested: finishing this step, then saving a checkpoint ...", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        end_time = deadline(datetime.now(), args.stop_at, args.minutes)
    except ValueError:
        sys.exit("--stop-at must look like 02:00")

    print(f"model {args.preset}: {n_params:,} parameters | train.bin {len(train_data):,} tokens | "
          f"val.bin {len(val_data):,} tokens")
    print(f"{tokens_per_step:,} tokens per step | {max_steps:,} steps = one pass over train.bin | "
          f"warm-up {warmup} steps | output folder {out}")
    print(f"will stop at {end_time:%Y-%m-%d %H:%M}" if end_time else "no time limit: stops at the last step",
          flush=True)

    @torch.no_grad()
    def validation_loss():
        model.eval()
        rng_eval = np.random.default_rng(0)          # same windows every time: curves are comparable
        losses = []
        for _ in range(args.eval_iters):
            x, y = sample_windows(val_data, rng_eval, args.batch, T)
            _, loss = model(torch.from_numpy(x), torch.from_numpy(y))
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    log_path = out / "log.csv"
    if not log_path.exists():
        log_path.write_text("step,tokens_seen,train_loss,val_loss,lr,time\n")

    def evaluate_and_log(train_loss, lr_now, show_sample):
        nonlocal best_val
        val = validation_loss()
        marker = ""
        if val < best_val:
            best_val = val
            atomic_save(model.state_dict(), out / "model_best.pt")
            marker = "  <- best so far"
        print(f"=== step {step:,}: train loss {train_loss:.3f} | validation loss {val:.3f}{marker}", flush=True)
        with open(log_path, "a") as f:
            f.write(f"{step},{tokens_seen},{train_loss:.4f},{val:.4f},{lr_now:.3e},"
                    f"{datetime.now().isoformat(timespec='seconds')}\n")
        if show_sample and tok is not None:
            prompt = PROMPTS[(step // max(1, args.eval_every)) % len(PROMPTS)]
            print("--- sample ---")
            print(sample_text(model, tok, cfg, prompt))
            print("--------------", flush=True)

    # ------------------------------------------------------------ the training loop
    model.train()
    rng = np.random.default_rng(args.seed + step)    # a different seed after each resume
    torch.manual_seed(args.seed + step)
    started = last_ckpt = window_start = time.time()
    window_steps = 0
    loss_sum = float("nan")
    lr_now = args.lr
    reason = "finished all steps"
    while step < max_steps:
        lr_now = lr_at(step, max_steps, args.lr, args.min_lr, warmup)
        for group in opt.param_groups:
            group["lr"] = lr_now
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(args.accum):
            x, y = sample_windows(train_data, rng, args.batch, T)
            _, loss = model(torch.from_numpy(x), torch.from_numpy(y))
            (loss / args.accum).backward()
            loss_sum += loss.item() / args.accum
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        step += 1
        tokens_seen += tokens_per_step
        window_steps += 1

        if not math.isfinite(loss_sum):
            sys.exit("the loss became NaN/inf: stopping. The last saved checkpoint is still good.")

        if step % args.log_every == 0:
            now = time.time()
            tps = window_steps * tokens_per_step / (now - window_start)
            remaining = (max_steps - step) * tokens_per_step / tps
            print(f"step {step:6,}/{max_steps:,} | loss {loss_sum:.3f} | lr {lr_now:.1e} | "
                  f"{tps:5.0f} tok/s | grad {grad_norm:.2f} | {hours_text(remaining)} left for the full pass",
                  flush=True)
            window_start, window_steps = now, 0

        if step % args.eval_every == 0:
            evaluate_and_log(loss_sum, lr_now, show_sample=True)
            window_start = time.time()

        if stop["flag"]:
            reason = "stop requested"
            break
        if end_time is not None and datetime.now() >= end_time:
            reason = "time window ended"
            break
        if time.time() - last_ckpt >= args.ckpt_minutes * 60:
            save_checkpoint(out, model, opt, step, tokens_seen, best_val, cfg, tokens_per_step)
            last_ckpt = time.time()
            print(f"(checkpoint saved at step {step:,})", flush=True)

    # ------------------------------------------------------------ the end of this session
    if step > 0 and step % args.eval_every != 0:
        evaluate_and_log(loss_sum, lr_now, show_sample=False)
    save_checkpoint(out, model, opt, step, tokens_seen, best_val, cfg, tokens_per_step)
    elapsed = time.time() - started
    print(f"\nsession over ({reason}). step {step:,}/{max_steps:,} | {tokens_seen / 1e6:.1f}M tokens seen | "
          f"best validation loss {best_val:.3f} | this session {hours_text(elapsed)}")
    if step >= max_steps:
        print("TRAINING COMPLETE. The final weights are in ckpt_last.pt; the best ones in model_best.pt.")
    else:
        print("Run the same command again to continue from the checkpoint.")


if __name__ == "__main__":
    main()
