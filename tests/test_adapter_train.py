import os

import numpy as np

from tgg.adapter.train import DEFAULTS, train
from tgg.eval.metrics import masked_mae
from tgg.tsfm import cache

N, D, H = 10, 32, 12
T = 400


def _data():
    rng = np.random.default_rng(0)
    steps = np.arange(T)
    X = (60 + 8 * np.sin(2 * np.pi * steps / 288)[:, None] + rng.normal(0, 1, (T, N))).astype(np.float32)
    M = np.ones((T, N), dtype=np.uint8)
    return X, M


def _fake_cache(tmp_path, X, M, t_by_split, seed=0):
    rng = np.random.default_rng(seed)
    for split, t in t_by_split.items():
        n = len(t)
        lb = t[:, None] - np.arange(11, -1, -1)[None, :]
        x_lb, m_lb = X[lb].transpose(0, 2, 1), M[lb].transpose(0, 2, 1).astype(bool)
        arrays = cache.create(os.path.join(str(tmp_path), split), n, N, D)
        arrays["t"][:] = t
        arrays["obs"][:] = m_lb.any(-1)
        arrays["s"][:] = cache.window_scale(x_lb, m_lb)
        arrays["loc"][:] = 0.0
        arrays["sd"][:] = 1.0
        arrays["h"][:] = rng.normal(0, 1, (n, N, D)).astype(np.float16)
        arrays["y0"][:] = np.repeat(X[t][:, :, None], H, axis=2).astype(np.float32)
        for a in arrays.values():
            a.flush()
    return str(tmp_path)


def _nbr():
    upper = np.triu(np.ones((N, N), dtype=bool), 1)
    import torch
    return torch.from_numpy(upper | upper.T | np.eye(N, dtype=bool))


def test_step0_val_mae_equals_zero_shot_exactly(tmp_path):
    X, M = _data()
    train_t = np.arange(20, 60, dtype=np.int64)
    val_t = np.arange(100, 140, dtype=np.int64)
    cache_dir = _fake_cache(tmp_path, X, M, {"train": train_t, "val": val_t})

    out = train({"cache_dir": cache_dir, "variant": "gat", "X": X, "M": M,
                 "train_ids": train_t, "nbr": _nbr(), "d": D, "eval_every": 1000, "max_steps": 1})

    future = val_t[:, None] + np.arange(1, H + 1)
    y = X[future].transpose(0, 2, 1)
    m = M[future].transpose(0, 2, 1).astype(bool)
    y0 = np.repeat(X[val_t][:, :, None], H, axis=2)
    zero_shot_mae = masked_mae(y, y0, m)

    assert round(out["history"][0]["val_mae"], 4) == round(zero_shot_mae, 4)


def test_overfits_a_tiny_training_set(tmp_path):
    X, M = _data()
    train_t = np.array([30, 40, 50, 60], dtype=np.int64)
    val_t = np.arange(100, 110, dtype=np.int64)
    cache_dir = _fake_cache(tmp_path, X, M, {"train": train_t, "val": val_t})

    out = train({"cache_dir": cache_dir, "variant": "gat", "X": X, "M": M, "train_ids": train_t,
                 "nbr": _nbr(), "d": D, "batch_windows": 4, "eval_every": 100,
                 "max_steps": 3000, "patience_evals": 1000})

    losses = [h["train_loss"] for h in out["history"] if h["train_loss"] is not None]
    assert losses[-1] < 0.5 * losses[0], losses                        # the loop can fit (guide C12)


def test_early_stopping_and_step_cap_are_respected(tmp_path):
    X, M = _data()
    train_t = np.arange(20, 60, dtype=np.int64)
    val_t = np.arange(100, 140, dtype=np.int64)
    cache_dir = _fake_cache(tmp_path, X, M, {"train": train_t, "val": val_t})

    out = train({"cache_dir": cache_dir, "variant": "mlp", "X": X, "M": M, "train_ids": train_t,
                 "nbr": _nbr(), "d": D, "eval_every": 5, "max_steps": 40, "patience_evals": 2})

    assert out["best_step"] <= 40
    evals = [h for h in out["history"] if h["step"] > 0]
    assert len(evals) <= 40 // 5
