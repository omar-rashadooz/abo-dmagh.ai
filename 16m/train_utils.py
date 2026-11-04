"""Small pure-python helpers for ch7_train.py (no torch here, so they are easy to test)."""
import math
from datetime import timedelta

import numpy as np


def sample_windows(data, rng, batch, block):
    """Pick `batch` random windows of `block` tokens. y is x shifted by one: the 'next token' answers."""
    starts = rng.integers(0, len(data) - block - 1, size=batch)
    x = np.stack([data[s:s + block] for s in starts]).astype(np.int64)
    y = np.stack([data[s + 1:s + 1 + block] for s in starts]).astype(np.int64)
    return x, y


def warmup_steps(max_steps):
    """Learning rate climbs for the first ~5% of training (between 10 and 300 steps)."""
    return max(10, min(300, int(0.05 * max_steps)))


def lr_at(step, max_steps, peak, floor, warmup):
    """Warm-up, then a smooth cosine slope from `peak` down to `floor`."""
    if step < warmup:
        return peak * (step + 1) / warmup
    if step >= max_steps:
        return floor
    progress = (step - warmup) / max(1, max_steps - warmup)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))


def parse_clock(text):
    """'02:00' -> (2, 0). Raises ValueError for anything else."""
    hour_text, minute_text = text.split(":")
    hour, minute = int(hour_text), int(minute_text)
    if not (0 <= hour < 24 and 0 <= minute < 60):
        raise ValueError(f"bad time: {text}")
    return hour, minute


def deadline(now, stop_at=None, minutes=None):
    """When must training stop? The earliest of: the next 'HH:MM' on the clock, and now + minutes."""
    candidates = []
    if stop_at:
        hour, minute = parse_clock(stop_at)
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        candidates.append(target)
    if minutes is not None:
        candidates.append(now + timedelta(minutes=minutes))
    return min(candidates) if candidates else None


def hours_text(seconds):
    hours = seconds / 3600
    return f"{hours:.1f} h" if hours >= 1 else f"{seconds / 60:.0f} min"
