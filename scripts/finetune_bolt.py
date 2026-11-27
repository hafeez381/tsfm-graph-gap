"""Full fine-tuning of Chronos-Bolt Small at one fraction, score it (guide C15).

Usage: python -m scripts.finetune_bolt --fraction 1.0 --mode smoke
       python -m scripts.finetune_bolt --fraction 1.0 --mode full --device cuda
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
from tgg.tsfm import bolt, finetune

H, L = 12, 12
DEFAULTS = {"lr": 1e-5, "wd": 0.01, "grad_clip": 1.0, "batch": 256, "eval_every": 500,
            "max_steps": 20000, "patience_evals": 5, "val_fraction": 0.2}
SMOKE_T, SMOKE_N = 600, 10


def _synthetic_data() -> dict:
    rng = np.random.default_rng(GLOBAL_SEED)
    steps = np.arange(SMOKE_T)
    X = 60 + 8 * np.sin(2 * np.pi * steps / 288)[:, None] + rng.normal(0, 2, (SMOKE_T, SMOKE_N))
    M = (rng.random((SMOKE_T, SMOKE_N)) >= 0.05).astype(np.uint8)
    X = (X * M).astype(np.float32)
    return {"X": X, "M": M, "X_filled": X, "tod": np.zeros(SMOKE_T, np.float32),
            "ts": (1_330_000_000 + 300 * steps).astype(np.int64)}


def _predict_median(pipeline, context: np.ndarray, batch: int = 256) -> np.ndarray:
    """(n, H) mph: pipeline.predict's 0.5 quantile, batched (no gradient, no embedding capture)."""
    out = []
    for i in range(0, len(context), batch):
        q = pipeline.predict(torch.from_numpy(context[i:i + batch]), prediction_length=H)
        out.append(q[:, finetune.MEDIAN_IDX, :].to(torch.float32).numpy())
    return np.concatenate(out)


def _predict_split(pipeline, X: np.ndarray, M: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    context, _, _ = finetune.contexts_and_targets(X, M, t, L, H)
    N = X.shape[1]
    y0 = _predict_median(pipeline, context).reshape(len(t), N, H)
    return t, y0


def run(fraction: float, mode: str, ds: str, data: str, results: str, device: str | None = None,
        seed: int = GLOBAL_SEED, cfg: dict | None = None) -> dict:
    t0 = time.time()
    seed_everything(seed)
    c = {**DEFAULTS, **(cfg or {})}
    if mode == "smoke":
        d = _synthetic_data()
        w = windows.make(SMOKE_T, L_long=288, H=H)
        data_dir = os.path.join(results, "_smoke_data_ft")
        os.makedirs(data_dir, exist_ok=True)
        np.savez(os.path.join(data_dir, f"{ds}.npz"), **d)
        np.savez(os.path.join(data_dir, f"{ds}_windows.npz"), **w)
        X, M = d["X"], d["M"]
        c.update({"eval_every": 2, "max_steps": 4, "patience_evals": 3, "batch": 8})
    else:
        data_dir = data
        dd = np.load(os.path.join(data_dir, f"{ds}.npz"))
        w = np.load(os.path.join(data_dir, f"{ds}_windows.npz"))
        X, M = dd["X"], dd["M"]

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = finetune.load(device)
    N = X.shape[1]

    train_ids = windows.fraction_block(w["train_t"], fraction)
    context_tr, y_tr, my_tr = finetune.contexts_and_targets(X, M, train_ids, L, H)
    x_raw_tr, m_raw_tr = np.where(np.isnan(context_tr), 0, context_tr), ~np.isnan(context_tr)
    keep = finetune.drop_flat_windows(x_raw_tr, m_raw_tr)
    n_dropped = int((~keep).sum())
    context_tr, y_tr, my_tr = context_tr[keep], y_tr[keep], my_tr[keep]

    rng = np.random.default_rng(seed)
    val_ids = w["val_t"]
    context_val, y_val, my_val = finetune.contexts_and_targets(X, M, val_ids, L, H)
    n_val_rows = len(context_val)
    val_sub = rng.choice(n_val_rows, size=max(1, int(round(c["val_fraction"] * n_val_rows))), replace=False)

    def masked_mae(y0: np.ndarray, y: np.ndarray, m: np.ndarray) -> float:
        if not m.any():
            raise AssertionError("no observed targets in this evaluation")
        return float(np.abs(y0 - y)[m].mean())

    opt = torch.optim.AdamW(pipeline.model.parameters(), lr=c["lr"], weight_decay=c["wd"])
    n_train = len(context_tr)

    pipeline.model.eval()
    with torch.no_grad():
        y0_init = _predict_median(pipeline, context_val[val_sub], batch=c["batch"])
    init_val_mae = masked_mae(y0_init, y_val[val_sub], my_val[val_sub])
    pipeline.model.train()

    history = [{"step": 0, "train_loss": None, "val_mae": init_val_mae}]
    best_val, best_step, since_best = init_val_mae, 0, 0
    best_state = {k: v.detach().clone() for k, v in pipeline.model.state_dict().items()}

    step = 0
    while step < c["max_steps"] and since_best < c["patience_evals"]:
        idx = rng.choice(n_train, size=min(c["batch"], n_train), replace=n_train < c["batch"])
        ctx = torch.from_numpy(context_tr[idx])
        tgt = torch.from_numpy(y_tr[idx])
        tgt_mask = torch.from_numpy(my_tr[idx])
        loss, _ = finetune.loss_and_preds(pipeline, ctx, tgt, tgt_mask)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pipeline.model.parameters(), c["grad_clip"])
        opt.step()
        step += 1

        if step % c["eval_every"] == 0:
            pipeline.model.eval()
            with torch.no_grad():
                y0_val = _predict_median(pipeline, context_val[val_sub], batch=c["batch"])
            val_mae = masked_mae(y0_val, y_val[val_sub], my_val[val_sub])
            pipeline.model.train()
            history.append({"step": step, "train_loss": loss.item(), "val_mae": val_mae})
            if val_mae < best_val:
                best_val, best_step, since_best = val_mae, step, 0
                best_state = {k: v.detach().clone() for k, v in pipeline.model.state_dict().items()}
            else:
                since_best += 1

    pipeline.model.load_state_dict(best_state)
    pipeline.model.eval()

    run_id = f"ft-bolt-L12-f{int(round(fraction * 100))}-s{seed}"
    run_dir = os.path.join(results, run_id)
    os.makedirs(run_dir, exist_ok=True)

    metrics = {}
    for split in ("val", "test"):
        t, y0 = _predict_split(pipeline, X, M, w[f"{split}_t"])
        path = os.path.join(run_dir, f"{split}_preds.npz")
        np.savez(path, t=t, yhat=y0)
        metrics[split] = score(path, ds, split, data_dir)

    with open(os.path.join(run_dir, "log.jsonl"), "w") as f:
        for row in history:
            f.write(json.dumps(row) + "\n")
    torch.save(best_state, os.path.join(run_dir, "best.pt"))

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = {"git_sha": sha, "run_id": run_id, "fraction": fraction, "seed": seed, "best_step": best_step,
            "best_val_mae": best_val, "n_train_sensor_windows": n_train, "n_dropped_flat_windows": n_dropped,
            "trained_params": sum(p.numel() for p in pipeline.model.parameters()), "frozen_params": 0,
            "wall_seconds": round(time.time() - t0, 2)}
    with open(os.path.join(run_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return {"run_id": run_id, "meta": meta, "val_metrics": metrics["val"], "test_metrics": metrics["test"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fraction", type=float, required=True)
    ap.add_argument("--mode", default="smoke", choices=["smoke", "full"])
    ap.add_argument("--ds", default="metr_la")
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    print(json.dumps(run(**vars(ap.parse_args())), indent=2, default=str))
