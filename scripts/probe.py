"""Ridge probe: does the pooled embedding carry the recent trend, or just the raw level? (guide C10)

Usage: python -m scripts.probe --mode smoke
       python -m scripts.probe --mode full --data data --cache cache
"""
import argparse
import json
import os

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from tgg.seed import GLOBAL_SEED, seed_everything

RIDGE_ALPHAS = (0.1, 1, 10, 100)
FIT_N_TRAIN, FIT_N_VAL = 200_000, 20_000
L = 12                                                                 # the last-hour change window


def trend_target(X: np.ndarray, M: np.ndarray, t: np.ndarray, sd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n, N) (x_t - x_{t-11}) / sd and (n, N) bool, defined only where both ends are observed."""
    both_obs = (M[t] == 1) & (M[t - (L - 1)] == 1)
    y = np.where(both_obs, (X[t] - X[t - (L - 1)]) / np.maximum(sd, 1e-6), np.nan)
    return y, both_obs


def _sample_rows(cache_dir: str, split: str, X: np.ndarray, M: np.ndarray, h_field: str,
                  n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """h, trend target and level target for n random (window, sensor) rows with both ends observed."""
    t = np.load(os.path.join(cache_dir, split, "t.npy"), mmap_mode="r")
    obs = np.asarray(np.load(os.path.join(cache_dir, split, "obs.npy"), mmap_mode="r"))
    h = np.load(os.path.join(cache_dir, split, f"{h_field}.npy"), mmap_mode="r")
    sd = np.asarray(np.load(os.path.join(cache_dir, split, "sd.npy"), mmap_mode="r"))
    t = np.asarray(t)
    y_trend, both_obs = trend_target(X, M, t, sd)
    keep = obs & both_obs
    wi, ni = np.nonzero(keep)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(wi), size=min(n, len(wi)), replace=False)
    wi, ni = wi[idx], ni[idx]
    return np.asarray(h[wi, ni]).astype(np.float32), y_trend[wi, ni].astype(np.float32), X[t[wi], ni].astype(np.float32)


def fit_and_score_best_alpha(h_train: np.ndarray, y_train: np.ndarray, h_val: np.ndarray,
                              y_val: np.ndarray, alphas=RIDGE_ALPHAS) -> dict:
    """Fit a ridge per alpha on train, score R^2 on validation, keep the best (guide: never on train)."""
    best = None
    for a in alphas:
        model = Ridge(alpha=a)
        model.fit(h_train, y_train)
        r2 = r2_score(y_val, model.predict(h_val))
        if best is None or r2 > best["r2"]:
            best = {"alpha": a, "r2": r2}
    return best


def run_probe(cache_dir: str, X: np.ndarray, M: np.ndarray, h_field: str = "h",
              n_train: int = FIT_N_TRAIN, n_val: int = FIT_N_VAL) -> dict:
    h_tr, y_tr, level_tr = _sample_rows(cache_dir, "train", X, M, h_field, n_train, seed=651)
    h_va, y_va, level_va = _sample_rows(cache_dir, "val", X, M, h_field, n_val, seed=652)
    trend = fit_and_score_best_alpha(h_tr, y_tr, h_va, y_va)
    level = fit_and_score_best_alpha(h_tr, level_tr, h_va, level_va)
    return {"trend_r2": trend["r2"], "trend_alpha": trend["alpha"],
            "level_r2": level["r2"], "level_alpha": level["alpha"]}


def run(ds: str, mode: str, data: str, cache: str, results: str) -> dict:
    seed_everything(GLOBAL_SEED)
    data_dir = os.path.join(cache, "_smoke_data") if mode == "smoke" else data
    d = np.load(os.path.join(data_dir, f"{ds}.npz"))
    X, M = d["X"], d["M"]

    probes = {
        "bolt-L12": (os.path.join(cache, "bolt-L12"), "h"),
        "bolt-L288-mean": (os.path.join(cache, "bolt-L288"), "h"),
        "bolt-L288-h_last": (os.path.join(cache, "bolt-L288"), "h_last"),
        "timesfm-L12": (os.path.join(cache, "timesfm-L12"), "h"),
    }
    n_train = 2000 if mode == "smoke" else FIT_N_TRAIN
    n_val = 500 if mode == "smoke" else FIT_N_VAL
    out = {name: run_probe(cache_dir, X, M, h_field, n_train, n_val)
           for name, (cache_dir, h_field) in probes.items() if os.path.exists(cache_dir)}

    for name, r in out.items():
        if r["level_r2"] >= r["trend_r2"]:
            r["warning"] = "level R^2 >= trend R^2: the embedding may not be scale-free as assumed"

    os.makedirs(os.path.join(results, "probe"), exist_ok=True)
    with open(os.path.join(results, "probe", "r2.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="metr_la")
    ap.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--data", default="data")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--results", default="results")
    print(json.dumps(run(**vars(ap.parse_args())), indent=2, default=str))
