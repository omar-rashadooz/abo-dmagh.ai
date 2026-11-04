"""Chapter 7 (GPU version): train abo-dmagh.ai on a GPU (Colab T4) or on the CPU.

This is YOUR GPU edit of ch7_train.py (device = cuda or cpu, tensors moved to the device),
plus four small changes:
  1. mixed precision (fp16 autocast + GradScaler): a T4 is MUCH faster in fp16 than in fp32. Use --no-amp to turn it off.
  2. a bigger micro-batch on the GPU: --batch 32 --accum 1 (still 8,192 tokens per step, so it resumes your CPU checkpoint).
  3. --mirror DIR: after every checkpoint, copy ckpt_last.pt / model_best.pt / log.csv to DIR (put Google Drive there),
     because a Colab session can disappear without warning.
  4. validation always uses the SAME windows as before (8 windows x --eval-iters), whatever --batch is,
     so the validation curve stays comparable with your laptop run.

    python ch7_train_gpu.py                                   # resumes from runs/abo-dmagh-16M/ckpt_last.pt
    python ch7_train_gpu.py --mirror /content/drive/MyDrive/abodmagh_runs --ckpt-minutes 10
    python ch7_train_gpu.py --minutes 180                     # stop (and save) after 3 hours
    python ch7_train_gpu.py --no-amp                          # full fp32, to compare

On Colab use --minutes, not --stop-at (Colab's clock is UTC, not Cairo time).
Press Ctrl+C once to stop cleanly (it saves a checkpoint first).
"""
import argparse
import json
import math
import os
import shutil
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

# ---- Device Configuration (GPU Support) ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "clean"
EVAL_BATCH = 8   # fixed on purpose: the validation windows must not change when --batch changes

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
    p.add_argument("--mirror", default=None, help="folder (e.g. on Google Drive) that receives a copy of every checkpoint")
    p.add_argument("--stop-at", default=None, help="clock time to stop at, e.g. 02:00")
    p.add_argument("--minutes", type=float, default=None, help="stop after this many minutes")
    p.add_argument("--max-steps", type=int, default=None, help="default: one pass over train.bin")
    p.add_argument("--batch", type=int, default=None, help="windows per micro-batch (default: 32 on GPU, 8 on CPU)")
    p.add_argument("--accum", type=int, default=None, help="micro-batches per step (default: 1 on GPU, 4 on CPU)")
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision (fp16) on the GPU")
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


def save_checkpoint(out, model, opt, scaler, step, tokens_seen, best_val, cfg, tokens_per_step):
    state = {
        "model": model.state_dict(), "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
        "step": step, "tokens_seen": tokens_seen, "best_val": best_val,
        "config": asdict(cfg), "tokens_per_step": tokens_per_step,
    }
    tmp = out / "ckpt_last.pt.tmp"
    torch.save(state, tmp)
    last, prev = out / "ckpt_last.pt", out / "ckpt_prev.pt"
    if last.exists():
        os.replace(last, prev)
    os.replace(tmp, last)


def mirror_files(out, mirror):
    """Copy the important files to `mirror` (Drive). A failure here must never stop the training."""
    if not mirror:
        return
    try:
        mirror.mkdir(parents=True, exist_ok=True)
        for name in ("ckpt_last.pt", "model_best.pt", "log.csv"):
            src = out / name
            if src.exists():
                tmp = mirror / (name + ".tmp")
                shutil.copy2(src, tmp)
                os.replace(tmp, mirror / name)
        print(f"(copied to {mirror})", flush=True)
    except Exception as error:
        print(f"WARNING: could not copy to {mirror}: {error}", flush=True)


def sample_text(model, tok, cfg, prompt, new_tokens=120, temperature=0.8, top_k=50):
    ids = tok.encode(prompt).ids
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    model.eval()
    with torch.no_grad():
        for _ in range(new_tokens):
            logits, _ = model(idx[:, -cfg.block_size:])
            logits = logits[:, -1, :].float() / temperature
            kth = torch.topk(logits, top_k).values[:, -1:]
            logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
    model.train()
    return tok.decode(idx[0].tolist())


def main():
    args = parse_args()
    on_gpu = device.type == "cuda"
    args.batch = args.batch or (32 if on_gpu else 8)
    args.accum = args.accum or (1 if on_gpu else 4)
    use_amp = on_gpu and not args.no_amp

    cfg = replace(PRESETS[args.preset], fast_attention=True)
    out = Path(args.out) if args.out else ROOT / "runs" / f"abo-dmagh-{args.preset}"
    out.mkdir(parents=True, exist_ok=True)
    mirror = Path(args.mirror) if args.mirror else None

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
    name = torch.cuda.get_device_name(0) if on_gpu else "cpu"
    print(f"Using device: {device} ({name}) | mixed precision (fp16): {use_amp}")
    model = GPT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    decay = [p for p in model.parameters() if p.dim() >= 2]       # weight matrices
    no_decay = [p for p in model.parameters() if p.dim() < 2]     # biases and LayerNorm
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def autocast():
        return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp)

    step, tokens_seen, best_val = 0, 0, float("inf")
    last = out / "ckpt_last.pt"
    if last.exists() and not args.fresh:
        ck = torch.load(last, map_location=device)
        if ck["config"] != asdict(cfg):
            sys.exit("the checkpoint was made with a different model size: use another --out or --fresh")
        if ck["tokens_per_step"] != tokens_per_step:
            sys.exit(f"batch*accum*block gives {tokens_per_step:,} tokens per step but the checkpoint used "
                     f"{ck['tokens_per_step']:,}: use the same product (e.g. --batch 32 --accum 1), or --fresh")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
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
    print(f"{tokens_per_step:,} tokens per step ({args.batch} x {args.accum} x {T}) | {max_steps:,} steps "
          f"= one pass over train.bin | warm-up {warmup} steps | output folder {out}")
    print(f"will stop at {end_time:%Y-%m-%d %H:%M}" if end_time else "no time limit: stops at the last step",
          flush=True)

    @torch.no_grad()
    def validation_loss():
        model.eval()
        rng_eval = np.random.default_rng(0)          # same windows every time: curves are comparable
        losses = []
        for _ in range(args.eval_iters):
            x, y = sample_windows(val_data, rng_eval, EVAL_BATCH, T)
            with autocast():
                _, loss = model(torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))
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
            with autocast():
                _, loss = model(torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))
            scaler.scale(loss / args.accum).backward()
            loss_sum += loss.item() / args.accum
        scaler.unscale_(opt)                          # so that clipping sees the real gradient size
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        scaler.step(opt)                              # skips the step by itself if the gradients overflowed
        scaler.update()
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
            save_checkpoint(out, model, opt, scaler, step, tokens_seen, best_val, cfg, tokens_per_step)
            mirror_files(out, mirror)
            last_ckpt = time.time()
            print(f"(checkpoint saved at step {step:,})", flush=True)
            window_start, window_steps = time.time(), 0

    # ------------------------------------------------------------ the end of this session
    if step > 0 and step % args.eval_every != 0:
        evaluate_and_log(loss_sum, lr_now, show_sample=False)
    save_checkpoint(out, model, opt, scaler, step, tokens_seen, best_val, cfg, tokens_per_step)
    mirror_files(out, mirror)
    elapsed = time.time() - started
    print(f"\nsession over ({reason}). step {step:,}/{max_steps:,} | {tokens_seen / 1e6:.1f}M tokens seen | "
          f"best validation loss {best_val:.3f} | this session {hours_text(elapsed)}")
    if step >= max_steps:
        print("TRAINING COMPLETE. The final weights are in ckpt_last.pt; the best ones in model_best.pt.")
    else:
        print("Run the same command again to continue from the checkpoint.")


if __name__ == "__main__":
    main()
