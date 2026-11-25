import os

import numpy as np
import torch

from scripts.sniff_test import (GAP_THRESHOLD_MPH, adapter_val_mae_60min, predict_60min_mae,
                                 run, zero_shot_val_mae)
from tgg.adapter.model import Adapter
from tgg.tsfm import cache

N, D, H = 6, 16, 12


def _fake_cache(tmp_path, X, M, t_by_split):
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
        arrays["h"][:] = np.random.default_rng(0).normal(0, 1, (n, N, D)).astype(np.float16)
        arrays["y0"][:] = np.repeat(X[t][:, :, None], H, axis=2).astype(np.float32)
        for a in arrays.values():
            a.flush()
    return str(tmp_path)


def _nbr():
    upper = np.triu(np.ones((N, N), dtype=bool), 1)
    return torch.from_numpy(upper | upper.T | np.eye(N, dtype=bool))


def test_zero_init_checkpoint_matches_zero_shot_at_60min(tmp_path):
    """Regression test: the adapter's reported number must use the same 60-minute slice as zero-shot,
    not an all-horizon average (the bug this script originally shipped with)."""
    rng = np.random.default_rng(1)
    T = 400
    X = (60 + rng.normal(0, 2, (T, N))).astype(np.float32)
    M = np.ones((T, N), dtype=np.uint8)
    val_t = np.arange(100, 140, dtype=np.int64)
    cache_dir = _fake_cache(tmp_path, X, M, {"val": val_t})

    zs = zero_shot_val_mae(cache_dir, val_t, X, M)
    zero_init_state = Adapter(d=D).state_dict()
    adapter_at_init = predict_60min_mae(zero_init_state, "mlp", cache_dir, val_t, X, M, _nbr(), d=D)
    assert zs == adapter_at_init                                      # W_g, b_g are zero: yhat == y0


def test_no_gap_is_a_no_go(tmp_path, monkeypatch):
    """If the zero-shot model is already accurate, the verdict must be no-go regardless of training."""
    rng = np.random.default_rng(2)
    T = 1500
    X = (60 + rng.normal(0, 0.01, (T, N))).astype(np.float32)          # near-perfectly predictable
    M = np.ones((T, N), dtype=np.uint8)
    from tgg.data import windows
    w = windows.make(T, L_long=288, H=H)
    cache_dir = _fake_cache(tmp_path, X, M, {"train": w["train_t"], "val": w["val_t"]})
    np.savez(tmp_path / "metr_la_graph.npz", nbr=_nbr().numpy())
    np.savez(tmp_path / "metr_la.npz", X=X, M=M, X_filled=X, tod=np.zeros(T, np.float32),
              ts=(1_330_000_000 + 300 * np.arange(T)).astype(np.int64))
    np.savez(tmp_path / "metr_la_windows.npz", **w)

    monkeypatch.setattr("scripts.sniff_test.N_TRAIN", min(50, len(w["train_t"])))
    monkeypatch.setattr("scripts.sniff_test.N_VAL", min(20, len(w["val_t"])))
    out = run(cache=cache_dir, ds="metr_la", data=str(tmp_path), mode="full", results=str(tmp_path / "out"))
    assert out["zero_shot_60min_val_mae"] < GAP_THRESHOLD_MPH
    assert out["verdict"] == "no-go"
