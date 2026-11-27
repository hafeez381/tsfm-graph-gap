"""Train PatchTST at one fraction, score it (guide C14).

Usage: python -m scripts.train_patchtst --fraction 1.0 --mode smoke
       python -m scripts.train_patchtst --fraction 1.0 --mode full
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tgg.adapter.train import DEFAULTS as ADAPTER_DEFAULTS
from tgg.baselines import patchtst
from tgg.data import windows
from tgg.eval.harness import score, targets
from tgg.seed import GLOBAL_SEED, seed_everything

SMOKE_T, SMOKE_N = 600, 10


def _synthetic_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(GLOBAL_SEED)
    steps = np.arange(SMOKE_T)
    raw = 60 + 8 * np.sin(2 * np.pi * steps / 288)[:, None] + rng.normal(0, 2, (SMOKE_T, SMOKE_N))
    M = (rng.random((SMOKE_T, SMOKE_N)) >= 0.05).astype(np.uint8)
    X = (raw * M).astype(np.float32)
    X_filled = X.copy()
    for n in range(SMOKE_N):
        last = 0.0
        for t in range(SMOKE_T):
            if M[t, n]:
                last = X_filled[t, n]
            else:
                X_filled[t, n] = last
    ts = (1_330_000_000 + 300 * steps).astype(np.int64)
    return X, M, X_filled.astype(np.float32), ts


class WindowIds(Dataset):
    def __init__(self, t: np.ndarray):
        self.t = t

    def __len__(self) -> int:
        return len(self.t)

    def __getitem__(self, i: int):
        return self.t[i]


def _make_collate(X_filled: np.ndarray, X: np.ndarray, M: np.ndarray):
    def collate(batch: list[int]):
        t = np.array(batch, dtype=np.int64)
        lb = t[:, None] - np.arange(patchtst.L - 1, -1, -1)[None, :]       # (n, L), oldest first
        x = X_filled[lb].transpose(0, 2, 1).astype(np.float32)             # (n, N, L)
        y, m = targets(X, M, t)
        return torch.from_numpy(x), torch.from_numpy(y.astype(np.float32)), torch.from_numpy(m), torch.from_numpy(t)
    return collate


def validate(model, loader) -> float:
    model.eval()
    total_abs, total_n = 0.0, 0
    with torch.no_grad():
        for x, y, m, _ in loader:
            if not m.any():
                continue
            yhat = patchtst.predict(model, x)
            total_abs += (yhat - y).abs()[m].sum().item()
            total_n += int(m.sum().item())
    model.train()
    if total_n == 0:
        raise AssertionError("no observed targets in the validation split")
    return total_abs / total_n


def run(fraction: float, mode: str, ds: str, data: str, results: str, seed: int = GLOBAL_SEED,
        cfg: dict | None = None) -> dict:
    t0 = time.time()
    seed_everything(seed)
    train_cfg = {**ADAPTER_DEFAULTS, **(cfg or {})}
    if mode == "smoke":
        X, M, X_filled, ts = _synthetic_data()
        w = windows.make(SMOKE_T, L_long=288, H=12)
        data_dir = os.path.join(results, "_smoke_data_patchtst")
        os.makedirs(data_dir, exist_ok=True)
        np.savez(os.path.join(data_dir, f"{ds}.npz"), X=X, M=M, X_filled=X_filled,
                 tod=np.zeros(SMOKE_T, np.float32), ts=ts)
        np.savez(os.path.join(data_dir, f"{ds}_windows.npz"), **w)
        train_cfg.update({"eval_every": 10, "max_steps": 50, "patience_evals": 3,
                           "batch_windows": min(ADAPTER_DEFAULTS["batch_windows"], SMOKE_T)})
    else:
        data_dir = data
        d = np.load(os.path.join(data_dir, f"{ds}.npz"))
        w = np.load(os.path.join(data_dir, f"{ds}_windows.npz"))
        X, M, X_filled, ts = d["X"], d["M"], d["X_filled"], d["ts"]

    model = patchtst.build()
    opt = torch.optim.Adam(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["wd"])

    train_ids = windows.fraction_block(w["train_t"], fraction)
    collate = _make_collate(X_filled, X, M)
    train_loader = DataLoader(WindowIds(train_ids), batch_size=train_cfg["batch_windows"], shuffle=True,
                               collate_fn=collate, drop_last=True)
    val_loader = DataLoader(WindowIds(w["val_t"]), batch_size=train_cfg["batch_windows"], shuffle=False,
                             collate_fn=collate)

    init_val_mae = validate(model, val_loader)
    history = [{"step": 0, "train_loss": None, "val_mae": init_val_mae}]
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_step, best_val, since_best = 0, init_val_mae, 0

    step, data_iter = 0, iter(train_loader)
    while step < train_cfg["max_steps"] and since_best < train_cfg["patience_evals"]:
        try:
            x, y, m, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x, y, m, _ = next(data_iter)
        if not m.any():
            continue
        yhat = patchtst.predict(model, x)
        loss = (yhat - y).abs()[m].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
        if step % train_cfg["eval_every"] == 0:
            val_mae = validate(model, val_loader)
            history.append({"step": step, "train_loss": loss.item(), "val_mae": val_mae})
            if val_mae < best_val:
                best_val, best_step, since_best = val_mae, step, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                since_best += 1
    model.load_state_dict(best_state)

    run_id = f"patchtst-L12-f{int(round(fraction * 100))}-s{seed}"
    run_dir = os.path.join(results, run_id)
    os.makedirs(run_dir, exist_ok=True)

    metrics = {}
    model.eval()
    for split in ("val", "test"):
        loader = DataLoader(WindowIds(w[f"{split}_t"]), batch_size=train_cfg["batch_windows"], shuffle=False,
                             collate_fn=collate)
        t_all, yhat_all = [], []
        with torch.no_grad():
            for x, _, _, t in loader:
                yhat_all.append(patchtst.predict(model, x).numpy())
                t_all.append(t.numpy())
        t_arr, yhat_arr = np.concatenate(t_all), np.concatenate(yhat_all)
        path = os.path.join(run_dir, f"{split}_preds.npz")
        np.savez(path, t=t_arr, yhat=yhat_arr)
        metrics[split] = score(path, ds, split, data_dir)

    with open(os.path.join(run_dir, "log.jsonl"), "w") as f:
        for row in history:
            f.write(json.dumps(row) + "\n")
    torch.save(best_state, os.path.join(run_dir, "best.pt"))

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = {"git_sha": sha, "run_id": run_id, "fraction": fraction, "seed": seed, "best_step": best_step,
            "best_val_mae": best_val, "trained_params": sum(p.numel() for p in model.parameters()),
            "frozen_params": 0, "wall_seconds": round(time.time() - t0, 2)}
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
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED)
    print(json.dumps(run(**vars(ap.parse_args())), indent=2, default=str))
