"""Zero-shot Chronos-2 forecasts with one group per window, L=12 and L=288, test split only, no
cache (guide C9).

Usage: python -m scripts.build_cache_chronos2 --mode smoke
       python -m scripts.build_cache_chronos2 --mode full --device cuda
"""
import argparse
import json
import os

import numpy as np

from tgg.data import windows
from tgg.eval.harness import score
from tgg.seed import GLOBAL_SEED, seed_everything
from tgg.tsfm import chronos2

H = 12
BATCH_WINDOWS = 32
SMOKE_T, SMOKE_N, SMOKE_WINDOWS = 600, 10, 8


def _synthetic_data() -> dict:
    """Synthetic speeds of the real shapes, for --mode smoke (mirrors scripts.skeleton.make_data)."""
    rng = np.random.default_rng(GLOBAL_SEED)
    steps = np.arange(SMOKE_T)
    X = 60 + 8 * np.sin(2 * np.pi * steps / 288)[:, None] + rng.normal(0, 2, (SMOKE_T, SMOKE_N))
    M = (rng.random((SMOKE_T, SMOKE_N)) >= 0.05).astype(np.uint8)
    X = (X * M).astype(np.float32)
    ts = (1_330_000_000 + 300 * steps).astype(np.int64)
    return {"X": X, "M": M, "X_filled": X, "tod": np.zeros(SMOKE_T, np.float32), "ts": ts}


def _windows_array(X: np.ndarray, M: np.ndarray, t: np.ndarray, L: int) -> np.ndarray:
    """(n, N, L) with NaN at masked steps (Decision 1), [..., -1] = step t, the most recent."""
    lb = t[:, None] - np.arange(L - 1, -1, -1)[None, :]
    x, m = X[lb].transpose(0, 2, 1), M[lb].transpose(0, 2, 1).astype(bool)
    return np.where(m, x, np.nan).astype(np.float32)


def run_L(pipeline, X: np.ndarray, M: np.ndarray, t: np.ndarray, L: int, ds: str, data_dir: str,
          results: str, batch_windows: int) -> dict:
    yhat = chronos2.run(pipeline, _windows_array(X, M, t, L), batch_size=batch_windows)
    preds_dir = os.path.join(results, f"chronos2-zs-L{L}")
    os.makedirs(preds_dir, exist_ok=True)
    preds_path = os.path.join(preds_dir, "test_preds.npz")
    np.savez(preds_path, t=t, yhat=yhat)
    return score(preds_path, ds, "test", data_dir)


def run(ds: str, mode: str, data: str, out: str, results: str, device: str | None,
        batch_windows: int) -> dict:
    seed_everything(GLOBAL_SEED)

    if mode == "smoke":
        d = _synthetic_data()
        w_full = windows.make(SMOKE_T, L_long=288, H=H)
        w = {k: (v[:SMOKE_WINDOWS] if k.endswith("_t") else v) for k, v in w_full.items()}
        data_dir = os.path.join(out, "_smoke_data")
        os.makedirs(data_dir, exist_ok=True)
        np.savez(os.path.join(data_dir, f"{ds}.npz"), **d)
        np.savez(os.path.join(data_dir, f"{ds}_windows.npz"), **w)
    else:
        data_dir = data
        d = np.load(os.path.join(data_dir, f"{ds}.npz"))
        w = np.load(os.path.join(data_dir, f"{ds}_windows.npz"))

    device = device or "cpu"
    pipeline = chronos2.load(device)
    return {f"zero_shot_test_metrics_L{L}": run_L(pipeline, d["X"], d["M"], w["test_t"], L, ds,
                                                    data_dir, results, batch_windows)
            for L in (12, 288)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="metr_la")
    ap.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="cache")
    ap.add_argument("--results", default="results")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-windows", type=int, default=BATCH_WINDOWS, dest="batch_windows")
    print(json.dumps(run(**vars(ap.parse_args())), indent=2, default=str))
