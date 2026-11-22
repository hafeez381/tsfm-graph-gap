"""Window ids per split, and few-shot fraction blocks.

A window is named by t, the index of its last lookback step. Its targets are steps t+1 .. t+H.
"""
import numpy as np


def split_bounds(T: int, split=(0.7, 0.1, 0.2)) -> tuple[int, int]:
    """Return (train_end, val_end): train steps are [0, train_end), validation [train_end, val_end), test [val_end, T)."""
    if abs(sum(split) - 1.0) > 1e-9:
        raise ValueError(f"split must sum to 1, got {split}")
    train_end = int(round(split[0] * T))
    val_end = int(round((split[0] + split[1]) * T))
    return train_end, val_end


def make(T: int, L_long: int = 288, H: int = 12, split=(0.7, 0.1, 0.2)) -> dict:
    """Window ids of each split.

    A window belongs to the split that holds all H of its target steps, so windows straddling a
    boundary are dropped. Every split keeps only t >= L_long - 1, so the longest lookback always
    exists and every model sees the same windows. A lookback may reach into the previous split.
    """
    train_end, val_end = split_bounds(T, split)
    t = np.arange(L_long - 1, T - H, dtype=np.int64)
    first, last = t + 1, t + H
    out = {
        "train_t": t[last < train_end],
        "val_t": t[(first >= train_end) & (last < val_end)],
        "test_t": t[first >= val_end],
        "train_end": train_end,
        "val_end": val_end,
    }
    for k in ("train_t", "val_t", "test_t"):
        if len(out[k]) == 0:
            raise ValueError(f"{k} is empty for T={T}, L_long={L_long}, H={H}")
        out[k] = out[k].astype(np.int32)
    return out


def fraction_block(train_t: np.ndarray, f: float) -> np.ndarray:
    """The most recent floor(f * n) training windows, a contiguous block just before validation."""
    if not 0 < f <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {f}")
    n = int(np.floor(f * len(train_t)))
    block = train_t[len(train_t) - n:]
    assert len(block) == n and (n == 0 or block[-1] == train_t[-1])
    return block
