"""Train one adapter variant on one cache at one fraction, score it, save its checkpoint (guide C12).

Usage: python -m scripts.train_adapter --cache bolt-L12 --variant gat --fraction 1.0 --mode smoke
       python -m scripts.train_adapter --cache bolt-L12 --variant gat --fraction 1.0 --mode full
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from tgg.adapter.data import CacheDataset
from tgg.adapter.model import Adapter, combine_mask, masks
from tgg.adapter.train import DEFAULTS, _collate, train
from tgg.data import windows
from tgg.eval.harness import score
from tgg.seed import GLOBAL_SEED, seed_everything
from tgg.tsfm import cache

N_ATTN_WINDOWS = 500
H = 12
SMOKE_T, SMOKE_N = 600, 10


def _synthetic_data_and_graph() -> tuple[dict, np.ndarray]:
    """Synthetic speeds and a random symmetric neighbour graph, for --mode smoke."""
    rng = np.random.default_rng(GLOBAL_SEED)
    steps = np.arange(SMOKE_T)
    X = 60 + 8 * np.sin(2 * np.pi * steps / 288)[:, None] + rng.normal(0, 2, (SMOKE_T, SMOKE_N))
    M = (rng.random((SMOKE_T, SMOKE_N)) >= 0.05).astype(np.uint8)
    X = (X * M).astype(np.float32)
    upper = np.triu(rng.random((SMOKE_N, SMOKE_N)) < 0.3, 1)
    nbr = upper | upper.T | np.eye(SMOKE_N, dtype=bool)
    return {"X": X, "M": M}, nbr


def make_fake_cache(cache_dir: str, X: np.ndarray, M: np.ndarray, w: dict, d: int = 512) -> None:
    """A fake cache with the real shapes, for development before job J6 lands (guide C12).
    y0 is the last observed lookback value repeated; h is random, uncorrelated with anything."""
    rng = np.random.default_rng(GLOBAL_SEED)
    N = X.shape[1]
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    for split in ("train", "val", "test"):
        t = w[f"{split}_t"]
        n = len(t)
        lb = t[:, None] - np.arange(11, -1, -1)[None, :]
        x_lb, m_lb = X[lb].transpose(0, 2, 1), M[lb].transpose(0, 2, 1).astype(bool)   # (n, N, 12)
        out_dir = os.path.join(cache_dir, split)
        arrays = cache.create(out_dir, n, N, d)
        arrays["t"][:] = t
        arrays["obs"][:] = m_lb.any(-1)
        arrays["s"][:] = cache.window_scale(x_lb, m_lb)
        arrays["loc"][:] = 0.0
        arrays["sd"][:] = 1.0
        arrays["h"][:] = rng.normal(0, 1, (n, N, d)).astype(np.float16)
        arrays["y0"][:] = np.repeat(X[t][:, :, None], H, axis=2).astype(np.float32)
        for a in arrays.values():
            a.flush()
        cache.write_manifest(out_dir, sha, "fake", names=list(arrays))


def _predict_split(model: Adapter, variant_mask: torch.Tensor, cache_dir: str, split: str,
                    X: np.ndarray, M: np.ndarray, batch_windows: int = 256) -> tuple[np.ndarray, np.ndarray]:
    ds = CacheDataset(cache_dir, split, X, M, None)
    loader = DataLoader(ds, batch_size=batch_windows, shuffle=False, collate_fn=_collate)
    model.eval()
    t_all, yhat_all = [], []
    with torch.no_grad():
        for batch in loader:
            mask = combine_mask(variant_mask, batch["obs"])
            yhat, _ = model(batch["h"], batch["y0"], batch["s"], mask)
            t_all.append(batch["t"].numpy())
            yhat_all.append(yhat.numpy())
    return np.concatenate(t_all), np.concatenate(yhat_all)


def _attention_on_fixed_windows(model: Adapter, variant_mask: torch.Tensor, cache_dir: str,
                                 X: np.ndarray, M: np.ndarray, n: int = N_ATTN_WINDOWS) -> dict:
    ds = CacheDataset(cache_dir, "test", X, M, None)
    ids = ds.t[: min(n, len(ds))]
    loader = DataLoader(CacheDataset(cache_dir, "test", X, M, ids), batch_size=len(ids) or 1,
                         shuffle=False, collate_fn=_collate)
    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        mask = combine_mask(variant_mask, batch["obs"])
        _, alpha = model(batch["h"], batch["y0"], batch["s"], mask)
    return {"t": batch["t"].numpy(), "alpha": alpha.numpy().astype(np.float16)}


def run(cache_name: str, variant: str, fraction: float, mode: str, ds: str, data: str, cache_root: str,
        results: str, seed: int = GLOBAL_SEED) -> dict:
    t0 = time.time()
    seed_everything(seed)
    if mode == "smoke":
        d, nbr_np = _synthetic_data_and_graph()
        X, M = d["X"], d["M"]
        w = windows.make(SMOKE_T, L_long=288, H=H)
        data_dir = os.path.join(cache_root, "_smoke_data")
        os.makedirs(data_dir, exist_ok=True)
        np.savez(os.path.join(data_dir, f"{ds}.npz"), X=X, M=M, X_filled=X,
                 tod=np.zeros(SMOKE_T, np.float32), ts=(1_330_000_000 + 300 * np.arange(SMOKE_T)).astype(np.int64))
        np.savez(os.path.join(data_dir, f"{ds}_windows.npz"), **w)
    else:
        data_dir = data
        dd = np.load(os.path.join(data_dir, f"{ds}.npz"))
        w = np.load(os.path.join(data_dir, f"{ds}_windows.npz"))
        g = np.load(os.path.join(data_dir, f"{ds}_graph.npz"))
        X, M, nbr_np = dd["X"], dd["M"], g["nbr"]
    nbr = torch.from_numpy(nbr_np)

    cache_dir = os.path.join(cache_root, cache_name)
    if mode == "smoke" and not os.path.exists(cache_dir):
        make_fake_cache(cache_dir, X, M, w)

    train_ids = windows.fraction_block(w["train_t"], fraction)
    cfg = {"cache_dir": cache_dir, "variant": variant, "X": X, "M": M, "train_ids": train_ids, "nbr": nbr}
    if mode == "smoke":
        cfg.update({"eval_every": 10, "max_steps": 50, "patience_evals": 3,
                    "batch_windows": min(DEFAULTS["batch_windows"], len(train_ids))})
    out = train(cfg)

    model = Adapter(d=DEFAULTS["d"], m=DEFAULTS["m"], heads=DEFAULTS["heads"], H=DEFAULTS["H"])
    model.load_state_dict(out["best_state"])
    variant_mask = masks(nbr)[variant]

    run_id = f"{variant}-{cache_name}-f{int(round(fraction * 100))}-s{seed}"
    run_dir = os.path.join(results, run_id)
    os.makedirs(run_dir, exist_ok=True)

    metrics = {}
    for split in ("val", "test"):
        t, yhat = _predict_split(model, variant_mask, cache_dir, split, X, M)
        path = os.path.join(run_dir, f"{split}_preds.npz")
        np.savez(path, t=t, yhat=yhat)
        metrics[split] = score(path, ds, split, data_dir)

    attn = _attention_on_fixed_windows(model, variant_mask, cache_dir, X, M)
    np.savez(os.path.join(run_dir, "alpha_test500.npz"), **attn)
    torch.save(out["best_state"], os.path.join(run_dir, "best.pt"))
    with open(os.path.join(run_dir, "log.jsonl"), "w") as f:
        for row in out["history"]:
            f.write(json.dumps(row) + "\n")

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = {"git_sha": sha, "run_id": run_id, "cache": cache_name, "variant": variant, "fraction": fraction,
            "seed": seed, "best_step": out["best_step"], "best_val_mae": out["best_val_mae"],
            "trained_params": sum(p.numel() for p in model.parameters()), "frozen_params": 0,
            "wall_seconds": round(time.time() - t0, 2)}
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return {"run_id": run_id, "meta": meta, "val_metrics": metrics["val"], "test_metrics": metrics["test"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, dest="cache_name")
    ap.add_argument("--variant", required=True, choices=["gat", "mlp", "full"])
    ap.add_argument("--fraction", type=float, required=True)
    ap.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--ds", default="metr_la")
    ap.add_argument("--data", default="data")
    ap.add_argument("--cache-root", default="cache", dest="cache_root")
    ap.add_argument("--results", default="results")
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    print(json.dumps(run(**vars(ap.parse_args())), indent=2, default=str))
