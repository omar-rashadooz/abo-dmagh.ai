"""sustained_speed.py - how fast does your laptop train when it has been working for a while?

The speed test in ch5_model.py lasts about a minute, so the CPU is still cool and running
at its turbo speed. A laptop under 6 hours of full load usually slows down (heat). This script
trains for several minutes and prints the speed of every minute, plus the CPU temperature.

Run (keep the laptop plugged in, nothing else heavy running):
    python sustained_speed.py            # 16M preset, 10 minutes
    python sustained_speed.py 28M 10     # preset name, minutes
"""
import glob
import sys
import time
from dataclasses import replace

import torch

from ch5_model import GPT, PRESETS

torch.manual_seed(0)


def cpu_temp():
    """Highest temperature (Celsius) reported by the Linux thermal zones, or None."""
    temps = []
    for path in glob.glob("/sys/class/thermal/thermal_zone*/temp"):
        try:
            with open(path) as f:
                temps.append(int(f.read().strip()) / 1000)
        except (OSError, ValueError):
            pass
    return max(temps) if temps else None


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "16M"
    minutes = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
    cfg = replace(PRESETS[name], fast_attention=True)
    batch = 8

    model = GPT(cfg)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randint(0, cfg.vocab_size, (batch, cfg.block_size))
    y = torch.randint(0, cfg.vocab_size, (batch, cfg.block_size))
    tokens_per_step = batch * cfg.block_size
    n_params = sum(p.numel() for p in model.parameters())

    def one_step():
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    print(f"preset {name} ({n_params:,} parameters), {minutes:g} minutes, "
          f"{torch.get_num_threads()} cpu threads")
    one_step()  # warm-up, not timed

    start = time.time()
    window_start = start
    window_tokens = 0
    total_tokens = 0
    rates = []
    while time.time() - start < minutes * 60:
        one_step()
        window_tokens += tokens_per_step
        total_tokens += tokens_per_step
        now = time.time()
        if now - window_start >= 60:
            rate = window_tokens / (now - window_start)
            rates.append(rate)
            temp = cpu_temp()
            temp_text = f"   cpu temp {temp:.0f} C" if temp is not None else ""
            print(f"minute {len(rates):2d}: {rate:6.0f} tokens/second{temp_text}", flush=True)
            window_start = now
            window_tokens = 0

    if not rates:
        print("the test was too short to report a full minute: use more minutes")
        return
    sustained = sum(rates[-3:]) / len(rates[-3:])
    print(f"\nfirst minute: {rates[0]:.0f} tokens/s   |   sustained (last minutes): {sustained:.0f} tokens/s")
    print(f"the laptop keeps {100 * sustained / rates[0]:.0f}% of its first-minute speed")
    print("\ntraining time at the SUSTAINED speed (6 hours per night):")
    for tokens in (100e6, 200e6, 300e6, 600e6):
        hours = tokens / sustained / 3600
        print(f"  {tokens / 1e6:4.0f}M tokens -> {hours:6.0f} hours = {hours / 6:5.1f} nights")


if __name__ == "__main__":
    main()
