"""Zero-shot Chronos-Bolt forecasts and the cache the adapters train on (guide C6, C7).

Usage: python -m scripts.build_cache --mode smoke
       python -m scripts.build_cache --mode full --L 12 --split train --device cuda
       python -m scripts.build_cache --mode full --L 288 --split test --device cuda   # guide C7
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
import torch

from tgg.data import windows
from tgg.eval.harness import score
from tgg.seed import GLOBAL_SEED, seed_everything
from tgg.tsfm import bolt, cache

H = 12
BATCH_SERIES = 4096
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


def _lookback(X: np.ndarray, M: np.ndarray, t: np.ndarray, L: int) -> tuple[np.ndarray, np.ndarray]:
    """Raw lookback (n, N, L) and its mask, [..., -1] = step t, the most recent (Decision 1)."""
    lb = t[:, None] - np.arange(L - 1, -1, -1)[None, :]            # (n, L), ascending t-L+1 .. t
    return X[lb].transpose(0, 2, 1), M[lb].transpose(0, 2, 1).astype(bool)


def build_split(pipeline, X: np.ndarray, M: np.ndarray, t: np.ndarray, out_dir: str, L: int,
                 encoder_id: str, h_last: bool = False, batch_series: int = BATCH_SERIES,
                 time_first: int = 1000) -> dict:
    """Run the frozen encoder over every window of `t`, write the cache of §6.2, return its arrays.
    h_last also stores the last-patch-token embedding beside the mean (guide C7)."""
    n, N = len(t), X.shape[1]
    arrays = cache.create(out_dir, n, N, bolt.D, h_last=h_last)
    b = max(1, batch_series // N)
    t0, timed_windows, timed_seconds = time.time(), 0, None
    for i in range(0, n, b):
        j = min(i + b, n)
        x_raw, m_raw = _lookback(X, M, t[i:j], L)                  # (j-i, N, L)
        contexts = np.where(m_raw, x_raw, np.nan).reshape(-1, L).astype(np.float32)   # Decision 1
        y0, h, loc, sd, hl = bolt.run(pipeline, torch.from_numpy(contexts))
        shape = (j - i, N)
        arrays["t"][i:j] = t[i:j]
        arrays["h"][i:j] = h.numpy().reshape(*shape, bolt.D).astype(np.float16)
        arrays["y0"][i:j] = y0.numpy().reshape(*shape, H)
        arrays["loc"][i:j] = loc.numpy().reshape(*shape)
        arrays["sd"][i:j] = sd.numpy().reshape(*shape)
        arrays["obs"][i:j] = m_raw.any(axis=-1)
        arrays["s"][i:j] = cache.window_scale(x_raw, m_raw)
        if h_last:
            arrays["h_last"][i:j] = hl.numpy().reshape(*shape, bolt.D).astype(np.float16)
        if timed_seconds is None and j >= min(time_first, n):
            timed_seconds, timed_windows = time.time() - t0, j
    for a in arrays.values():
        a.flush()
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    manifest = cache.write_manifest(out_dir, sha, encoder_id, names=list(arrays))
    timed_seconds = timed_seconds if timed_seconds is not None else 0.0
    timing = {"windows": timed_windows, "seconds": round(timed_seconds, 4),
              "windows_per_second": round(timed_windows / timed_seconds, 2) if timed_seconds > 0 else None}
    return {"manifest": manifest, "timing": timing, "arrays": arrays}


def run(encoder: str, L_: int, ds: str, split: str, mode: str, data: str, out: str, results: str,
        device: str | None, batch_series: int) -> dict:
    if encoder != "bolt" or L_ not in (12, 288):
        raise NotImplementedError("C6/C7 cover only the bolt-L12 and bolt-L288 caches")
    seed_everything(GLOBAL_SEED)
    encoder_id = f"bolt-L{L_}"
    h_last = L_ == 288                                             # guide §6.2: h_last only for bolt-L288

    if mode == "smoke":
        d = _synthetic_data()
        w_full = windows.make(SMOKE_T, L_long=288, H=H)
        # Truncate every split's window ids to what the smoke run actually scores, so score()'s
        # assert_common_targets (the full split's ids) matches what build_split below writes.
        w = {k: (v[:SMOKE_WINDOWS] if k.endswith("_t") else v) for k, v in w_full.items()}
        data_dir = os.path.join(out, "_smoke_data")
        os.makedirs(data_dir, exist_ok=True)
        np.savez(os.path.join(data_dir, f"{ds}.npz"), **d)
        np.savez(os.path.join(data_dir, f"{ds}_windows.npz"), **w)
    else:
        data_dir = data
        d = np.load(os.path.join(data_dir, f"{ds}.npz"))
        w = np.load(os.path.join(data_dir, f"{ds}_windows.npz"))

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = bolt.load(device)
    splits = ("train", "val", "test") if split == "all" else (split,)
    out_dirs = {s: os.path.join(out, encoder_id, s) for s in splits}

    out_results = {}
    for s in splits:
        out_results[s] = build_split(pipeline, d["X"], d["M"], w[f"{s}_t"], out_dirs[s], L_,
                                      encoder_id, h_last, batch_series)

    if "test" in splits:
        preds_dir = os.path.join(results, f"bolt-zs-L{L_}")
        os.makedirs(preds_dir, exist_ok=True)
        preds_path = os.path.join(preds_dir, "test_preds.npz")
        arrays = out_results["test"]["arrays"]
        np.savez(preds_path, t=np.asarray(arrays["t"]), yhat=np.asarray(arrays["y0"]))
        out_results["zero_shot_test_metrics"] = score(preds_path, ds, "test", data_dir)

    for s in splits:
        out_results[s].pop("arrays")                              # memmaps are not JSON-serialisable
    return out_results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="bolt", choices=["bolt"])
    ap.add_argument("--L", type=int, default=12, dest="L_", choices=[12, 288])
    ap.add_argument("--ds", default="metr_la")
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="cache")
    ap.add_argument("--results", default="results")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-series", type=int, default=BATCH_SERIES, dest="batch_series")
    print(json.dumps(run(**vars(ap.parse_args())), indent=2, default=str))
