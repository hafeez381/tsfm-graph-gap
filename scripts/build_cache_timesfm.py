"""Zero-shot TimesFM 2.5 forecasts, its PCA-reduced cache at L=12, and the L=288 zero-shot-only
comparison (guide C8).

Usage: python -m scripts.build_cache_timesfm --mode smoke
       python -m scripts.build_cache_timesfm --mode full --device cuda
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np

from tgg.baselines.naive import assert_train_only_fit
from tgg.data import windows
from tgg.eval.harness import score
from tgg.seed import GLOBAL_SEED, seed_everything
from tgg.tsfm import cache
from tgg.tsfm import timesfm as tfm

H = 12
BATCH_SERIES = 1024
SMOKE_T, SMOKE_N, SMOKE_WINDOWS, SMOKE_PCA_N = 600, 10, 8, 200


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
    lb = t[:, None] - np.arange(L - 1, -1, -1)[None, :]
    return X[lb].transpose(0, 2, 1), M[lb].transpose(0, 2, 1).astype(bool)


def _nan_rows(x_raw: np.ndarray, m_raw: np.ndarray) -> np.ndarray:
    """(rows, L) raw lookback flattened over (window, sensor), with NaN at missing steps (Decision 1)."""
    rows = np.where(m_raw, x_raw, np.nan).reshape(-1, x_raw.shape[-1])
    return rows


def fit_pca_on_sample(model, X: np.ndarray, M: np.ndarray, train_t: np.ndarray, train_end: int,
                       n_fit: int, n_components: int = tfm.D, L: int = 12) -> object:
    """Sample n_fit (window, sensor) rows from training windows only (assert_train_only_fit) and
    fit the PCA on their raw TimesFM embeddings (guide C8)."""
    rng = np.random.default_rng(GLOBAL_SEED)
    N = X.shape[1]
    n_windows = min(len(train_t), -(-n_fit // N) + 1)
    sample_t = rng.choice(train_t, size=n_windows, replace=False)
    assert_train_only_fit(sample_t, train_end)
    x_raw, m_raw = _lookback(X, M, sample_t, L)
    rows = _nan_rows(x_raw, m_raw)
    obs = m_raw.any(axis=-1).reshape(-1)
    rows = rows[obs][:n_fit]
    _, h_raw = tfm.run(model, list(rows), horizon=H)
    return tfm.fit_pca(h_raw, n_components=n_components)


def build_split(model, pca, X: np.ndarray, M: np.ndarray, t: np.ndarray, out_dir: str, d: int = tfm.D,
                 batch_series: int = BATCH_SERIES, time_first: int = 1000) -> dict:
    """Run the frozen TimesFM encoder over every window of `t` at L=12, write the PCA-reduced cache."""
    n, N = len(t), X.shape[1]
    arrays = cache.create(out_dir, n, N, d)
    b = max(1, batch_series // N)
    t0, timed_windows, timed_seconds = time.time(), 0, None
    for i in range(0, n, b):
        j = min(i + b, n)
        x_raw, m_raw = _lookback(X, M, t[i:j], 12)
        rows = _nan_rows(x_raw, m_raw)
        obs_flat = m_raw.any(axis=-1).reshape(-1)
        shape = (j - i, N)

        y0_flat = np.zeros((rows.shape[0], H), dtype=np.float32)
        h_flat = np.zeros((rows.shape[0], d), dtype=np.float32)
        valid = np.flatnonzero(obs_flat)
        if len(valid):
            y0_valid, h_raw_valid = tfm.run(model, list(rows[valid]), horizon=H)
            y0_flat[valid] = y0_valid
            h_flat[valid] = tfm.apply_pca(pca, h_raw_valid)

        arrays["t"][i:j] = t[i:j]
        arrays["h"][i:j] = h_flat.reshape(*shape, d).astype(np.float16)
        arrays["y0"][i:j] = y0_flat.reshape(*shape, H)
        arrays["loc"][i:j] = 0.0                                    # TimesFM has no cached loc/scale
        arrays["sd"][i:j] = 1.0
        arrays["obs"][i:j] = obs_flat.reshape(*shape)
        arrays["s"][i:j] = cache.window_scale(x_raw, m_raw)
        if timed_seconds is None and j >= min(time_first, n):
            timed_seconds, timed_windows = time.time() - t0, j
    for a in arrays.values():
        a.flush()
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    manifest = cache.write_manifest(out_dir, sha, "timesfm-L12", names=list(arrays))
    timed_seconds = timed_seconds if timed_seconds is not None else 0.0
    timing = {"windows": timed_windows, "seconds": round(timed_seconds, 4),
              "windows_per_second": round(timed_windows / timed_seconds, 2) if timed_seconds > 0 else None}
    return {"manifest": manifest, "timing": timing, "arrays": arrays}


def zero_shot_288(model, X: np.ndarray, M: np.ndarray, t: np.ndarray, batch_series: int = BATCH_SERIES) -> np.ndarray:
    """Zero-shot point forecasts at L=288, test split only, no cache (guide C8)."""
    n, N = len(t), X.shape[1]
    b = max(1, batch_series // N)
    y0 = np.zeros((n, N, H), dtype=np.float32)
    for i in range(0, n, b):
        j = min(i + b, n)
        x_raw, m_raw = _lookback(X, M, t[i:j], 288)
        rows = _nan_rows(x_raw, m_raw)
        obs_flat = m_raw.any(axis=-1).reshape(-1)
        y0_flat = np.zeros((rows.shape[0], H), dtype=np.float32)
        valid = np.flatnonzero(obs_flat)
        if len(valid):
            y0_valid, _ = tfm.run(model, list(rows[valid]), horizon=H)
            y0_flat[valid] = y0_valid
        y0[i:j] = y0_flat.reshape(j - i, N, H)
    return y0


def run(ds: str, mode: str, data: str, out: str, results: str,
        batch_series: int, pca_fit_n: int) -> dict:
    seed_everything(GLOBAL_SEED)

    if mode == "smoke":
        d = _synthetic_data()
        w_full = windows.make(SMOKE_T, L_long=288, H=H)
        w = {k: (v[:SMOKE_WINDOWS] if k.endswith("_t") else v) for k, v in w_full.items()}
        data_dir = os.path.join(out, "_smoke_data")
        os.makedirs(data_dir, exist_ok=True)
        np.savez(os.path.join(data_dir, f"{ds}.npz"), **d)
        np.savez(os.path.join(data_dir, f"{ds}_windows.npz"), **w)
        pca_fit_n = min(pca_fit_n, SMOKE_PCA_N)
    else:
        data_dir = data
        d = np.load(os.path.join(data_dir, f"{ds}.npz"))
        w = np.load(os.path.join(data_dir, f"{ds}_windows.npz"))

    model12 = tfm.load(max_context=12)
    pca = fit_pca_on_sample(model12, d["X"], d["M"], w["train_t"], int(w["train_end"]), pca_fit_n)
    pca_variance = float(pca.explained_variance_ratio_.sum())
    n_components = pca.n_components_                                 # clamped to the sample's rank if small

    out_results = {"pca_explained_variance": pca_variance}
    for s in ("train", "val", "test"):
        out_dir = os.path.join(out, "timesfm-L12", s)
        out_results[s] = build_split(model12, pca, d["X"], d["M"], w[f"{s}_t"], out_dir,
                                      n_components, batch_series)

    preds_dir = os.path.join(results, "timesfm-zs-L12")
    os.makedirs(preds_dir, exist_ok=True)
    preds_path = os.path.join(preds_dir, "test_preds.npz")
    arrays = out_results["test"]["arrays"]
    np.savez(preds_path, t=np.asarray(arrays["t"]), yhat=np.asarray(arrays["y0"]))
    out_results["zero_shot_test_metrics_L12"] = score(preds_path, ds, "test", data_dir)
    for s in ("train", "val", "test"):
        out_results[s].pop("arrays")

    model288 = tfm.load(max_context=288)
    y0_288 = zero_shot_288(model288, d["X"], d["M"], w["test_t"], batch_series)
    preds_dir_288 = os.path.join(results, "timesfm-zs-L288")
    os.makedirs(preds_dir_288, exist_ok=True)
    preds_path_288 = os.path.join(preds_dir_288, "test_preds.npz")
    np.savez(preds_path_288, t=w["test_t"], yhat=y0_288)
    out_results["zero_shot_test_metrics_L288"] = score(preds_path_288, ds, "test", data_dir)
    return out_results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="metr_la")
    ap.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="cache")
    ap.add_argument("--results", default="results")
    ap.add_argument("--batch-series", type=int, default=BATCH_SERIES, dest="batch_series")
    ap.add_argument("--pca-fit-n", type=int, default=tfm.PCA_FIT_N, dest="pca_fit_n")
    print(json.dumps(run(**vars(ap.parse_args())), indent=2, default=str))
