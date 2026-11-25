"""Sniff test: does the frozen zero-shot model leave a gap a GAT adapter can close, on a small real
sample, before building the full pipeline (guide C6a)?

Usage: python -m scripts.sniff_test --mode smoke
       python -m scripts.sniff_test --mode full --cache cache/bolt-L12 --data data
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.train_adapter import make_fake_cache
from tgg.adapter.data import CacheDataset
from tgg.adapter.model import Adapter, combine_mask, masks
from tgg.adapter.train import DEFAULTS, _collate, train
from tgg.data import windows
from tgg.eval.metrics import masked_mae
from tgg.seed import GLOBAL_SEED, seed_everything

H_60MIN = 12                                                          # index into the 12-step horizon
N_TRAIN, N_VAL = 800, 200
GAP_THRESHOLD_MPH = 3.83                                              # Graph WaveNet's 3.53 + 0.3 (guide C6a)
MIN_GAT_IMPROVEMENT_MPH = 0.05
MAX_STEPS = 2000
SMOKE_T, SMOKE_N = 2000, 10


def _synthetic_data_and_graph() -> tuple[dict, np.ndarray]:
    rng = np.random.default_rng(GLOBAL_SEED)
    steps = np.arange(SMOKE_T)
    X = 60 + 8 * np.sin(2 * np.pi * steps / 288)[:, None] + rng.normal(0, 2, (SMOKE_T, SMOKE_N))
    M = (rng.random((SMOKE_T, SMOKE_N)) >= 0.05).astype(np.uint8)
    X = (X * M).astype(np.float32)
    upper = np.triu(rng.random((SMOKE_N, SMOKE_N)) < 0.3, 1)
    nbr = upper | upper.T | np.eye(SMOKE_N, dtype=bool)
    return {"X": X, "M": M}, nbr


def zero_shot_val_mae(cache_dir: str, val_ids: np.ndarray, X: np.ndarray, M: np.ndarray) -> float:
    """Masked MAE at 60 minutes of the cached zero-shot y0 against the raw targets, on exactly val_ids."""
    ds = CacheDataset(cache_dir, "val", X, M, val_ids)
    batch = next(iter(DataLoader(ds, batch_size=len(ds), shuffle=False, collate_fn=_collate)))
    y, y0, m = batch["y"].numpy(), batch["y0"].numpy(), batch["m"].numpy()
    return masked_mae(y[:, :, H_60MIN - 1:H_60MIN], y0[:, :, H_60MIN - 1:H_60MIN], m[:, :, H_60MIN - 1:H_60MIN])


def predict_60min_mae(state: dict, variant: str, cache_dir: str, val_ids: np.ndarray,
                       X: np.ndarray, M: np.ndarray, nbr: torch.Tensor, d: int = DEFAULTS["d"]) -> float:
    """60-minute masked MAE of the given checkpoint's predictions on exactly val_ids."""
    model = Adapter(d=d, m=DEFAULTS["m"], heads=DEFAULTS["heads"], H=DEFAULTS["H"])
    model.load_state_dict(state)
    model.eval()
    variant_mask = masks(nbr)[variant]
    ds = CacheDataset(cache_dir, "val", X, M, val_ids)
    batch = next(iter(DataLoader(ds, batch_size=len(ds), shuffle=False, collate_fn=_collate)))
    with torch.no_grad():
        mask = combine_mask(variant_mask, batch["obs"])
        yhat, _ = model(batch["h"], batch["y0"], batch["s"], mask)
    y, m = batch["y"].numpy(), batch["m"].numpy()
    yhat = yhat.numpy()
    sl = slice(H_60MIN - 1, H_60MIN)
    return masked_mae(y[:, :, sl], yhat[:, :, sl], m[:, :, sl])


def adapter_val_mae_60min(cache_dir: str, variant: str, train_ids: np.ndarray, val_ids: np.ndarray,
                           X: np.ndarray, M: np.ndarray, nbr: torch.Tensor, max_steps: int = MAX_STEPS,
                           d: int = DEFAULTS["d"]) -> dict:
    """Trains `variant` on train_ids for at most max_steps, selected on the whole-horizon validation
    MAE (as usual, Decision 8), then reports the best checkpoint's 60-minute MAE on val_ids, exactly
    comparable to zero_shot_val_mae (guide C6a compares both at 60 minutes)."""
    out = train({"cache_dir": cache_dir, "variant": variant, "X": X, "M": M, "train_ids": train_ids,
                 "val_ids": val_ids, "nbr": nbr, "max_steps": max_steps, "eval_every": 100,
                 "patience_evals": max_steps, "d": d})                # no early stop within the budget
    out["val_mae_60min"] = predict_60min_mae(out["best_state"], variant, cache_dir, val_ids, X, M, nbr, d)
    return out


def run(cache: str, ds: str, data: str, mode: str, results: str) -> dict:
    seed_everything(GLOBAL_SEED)
    if mode == "smoke":
        d, nbr_np = _synthetic_data_and_graph()
        X, M = d["X"], d["M"]
        w = windows.make(SMOKE_T, L_long=288, H=12)
        n_train, n_val = min(N_TRAIN, len(w["train_t"])), min(N_VAL, len(w["val_t"]))
        max_steps = 100
        if not os.path.exists(cache):
            make_fake_cache(cache, X, M, w)
    else:
        dd = np.load(os.path.join(data, f"{ds}.npz"))
        w = np.load(os.path.join(data, f"{ds}_windows.npz"))
        g = np.load(os.path.join(data, f"{ds}_graph.npz"))
        X, M, nbr_np = dd["X"], dd["M"], g["nbr"]
        n_train, n_val, max_steps = N_TRAIN, N_VAL, MAX_STEPS
    import torch
    nbr = torch.from_numpy(nbr_np)

    mini_train_t = w["train_t"][-n_train:]                           # the 800 (or fewer) windows before val
    mini_val_t = w["val_t"][:n_val]                                   # the first 200 (or fewer) val windows
    d_width = int(np.load(os.path.join(cache, "val", "h.npy"), mmap_mode="r").shape[-1])

    zs_mae = zero_shot_val_mae(cache, mini_val_t, X, M)
    gat = adapter_val_mae_60min(cache, "gat", mini_train_t, mini_val_t, X, M, nbr, max_steps, d_width)
    mlp = adapter_val_mae_60min(cache, "mlp", mini_train_t, mini_val_t, X, M, nbr, max_steps, d_width)

    gap_exists = zs_mae >= GAP_THRESHOLD_MPH
    gat_improvement = zs_mae - gat["val_mae_60min"]
    mlp_improvement = zs_mae - mlp["val_mae_60min"]
    gat_improves = gat_improvement >= MIN_GAT_IMPROVEMENT_MPH
    verdict = "go" if gap_exists and gat_improves else "no-go"
    out = {
        "verdict": verdict,
        "zero_shot_60min_val_mae": zs_mae,
        "gap_threshold_mph": GAP_THRESHOLD_MPH,
        "gap_exists": gap_exists,
        "gat_60min_val_mae": gat["val_mae_60min"],
        "mlp_60min_val_mae": mlp["val_mae_60min"],
        "gat_improvement_mph": gat_improvement,
        "gat_improves_enough": gat_improves,
        "mlp_improvement_mph": mlp_improvement,
        "note": ("neither adapter improves despite a gap: check the training loop on the fake cache"
                  if gap_exists and not gat_improves and mlp_improvement < MIN_GAT_IMPROVEMENT_MPH
                  else None),
    }
    os.makedirs(results, exist_ok=True)
    with open(os.path.join(results, "verdict.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/bolt-L12")
    ap.add_argument("--ds", default="metr_la")
    ap.add_argument("--data", default="data")
    ap.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--results", default="results/sniff")
    print(json.dumps(run(**vars(ap.parse_args())), indent=2, default=str))
