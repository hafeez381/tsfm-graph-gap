"""Train Graph WaveNet at one fraction, score it (guide C13).

Usage: python -m scripts.train_gwn --fraction 1.0 --mode smoke
       python -m scripts.train_gwn --fraction 1.0 --mode full
"""
import argparse
import json
import os
import subprocess
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tgg.baselines import gwn
from tgg.data import windows
from tgg.eval.harness import score, targets
from tgg.seed import GLOBAL_SEED, seed_everything

SMOKE_T, SMOKE_N = 600, 10


def _forward_fill(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    out = X.copy()
    for n in range(X.shape[1]):
        last = 0.0
        for t in range(X.shape[0]):
            if M[t, n]:
                last = out[t, n]
            else:
                out[t, n] = last
    return out


def _synthetic_data_and_graph() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Synthetic speeds, mask, forward-filled speeds, time of day and a random symmetric graph."""
    rng = np.random.default_rng(GLOBAL_SEED)
    steps = np.arange(SMOKE_T)
    raw = 60 + 8 * np.sin(2 * np.pi * steps / 288)[:, None] + rng.normal(0, 2, (SMOKE_T, SMOKE_N))
    M = (rng.random((SMOKE_T, SMOKE_N)) >= 0.05).astype(np.uint8)
    X = (raw * M).astype(np.float32)
    X_filled = _forward_fill(X, M).astype(np.float32)
    tod = ((steps % 288) / 288.0).astype(np.float32)
    upper = np.triu(rng.random((SMOKE_N, SMOKE_N)) < 0.3, 1)
    A = upper.astype(np.float32)
    return X, M, X_filled, tod, A


class WindowIds(Dataset):
    def __init__(self, t: np.ndarray):
        self.t = t

    def __len__(self) -> int:
        return len(self.t)

    def __getitem__(self, i: int):
        return self.t[i]


def _make_collate(X_filled: np.ndarray, tod: np.ndarray, X: np.ndarray, M: np.ndarray, scaler: gwn.Scaler):
    def collate(batch: list[int]):
        t = np.array(batch, dtype=np.int64)
        x = gwn.make_input(X_filled, tod, t, scaler)
        y, m = targets(X, M, t)
        return (torch.from_numpy(x), torch.from_numpy(y.astype(np.float32)),
                torch.from_numpy(m), torch.from_numpy(t))
    return collate


def _run_epoch(model, edge_index, edge_weight, loader, scaler, opt=None) -> float:
    """One pass over `loader`. Trains if `opt` is given, else returns the masked validation MAE."""
    model.train(opt is not None)
    total_abs, total_n = 0.0, 0
    for x, y, m, _ in loader:
        if not m.any():
            continue
        if opt is not None:
            yhat = gwn.predict(model, edge_index, edge_weight, x, scaler)
            loss = (yhat - y).abs()[m].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_abs += loss.item() * int(m.sum())
        else:
            with torch.no_grad():
                yhat = gwn.predict(model, edge_index, edge_weight, x, scaler)
            total_abs += (yhat - y).abs()[m].sum().item()
        total_n += int(m.sum().item())
    if total_n == 0:
        raise AssertionError("no observed targets in this pass")
    return total_abs / total_n


def _predict_split(model, edge_index, edge_weight, scaler, loader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    t_all, yhat_all = [], []
    with torch.no_grad():
        for x, _, _, t in loader:
            yhat = gwn.predict(model, edge_index, edge_weight, x, scaler)
            t_all.append(t.numpy())
            yhat_all.append(yhat.numpy())
    return np.concatenate(t_all), np.concatenate(yhat_all)


def run(fraction: float, mode: str, ds: str, data: str, results: str, seed: int = GLOBAL_SEED,
        cfg: dict | None = None) -> dict:
    t0 = time.time()
    seed_everything(seed)
    c = {**gwn.DEFAULTS, **(cfg or {})}
    if mode == "smoke":
        X, M, X_filled, tod, A = _synthetic_data_and_graph()
        w = windows.make(SMOKE_T, L_long=288, H=12)
        data_dir = os.path.join(results, "_smoke_data_gwn")
        os.makedirs(data_dir, exist_ok=True)
        np.savez(os.path.join(data_dir, f"{ds}.npz"), X=X, M=M, X_filled=X_filled, tod=tod,
                 ts=(1_330_000_000 + 300 * np.arange(SMOKE_T)).astype(np.int64))
        np.savez(os.path.join(data_dir, f"{ds}_windows.npz"), **w)
        c.update({"max_epochs": 3, "patience_epochs": 3, "batch_windows": 4})
    else:
        data_dir = data
        d = np.load(os.path.join(data_dir, f"{ds}.npz"))
        w = np.load(os.path.join(data_dir, f"{ds}_windows.npz"))
        g = np.load(os.path.join(data_dir, f"{ds}_graph.npz"))
        X, M, X_filled, tod, A = d["X"], d["M"], d["X_filled"], d["tod"], g["A"]

    N = X.shape[1]
    scaler = gwn.Scaler.fit(X_filled, int(w["train_end"]))
    model, edge_index, edge_weight = gwn.build(N, A, c)
    opt = torch.optim.Adam(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])

    train_ids = windows.fraction_block(w["train_t"], fraction)
    collate = _make_collate(X_filled, tod, X, M, scaler)
    train_loader = DataLoader(WindowIds(train_ids), batch_size=c["batch_windows"], shuffle=True,
                               collate_fn=collate, drop_last=True)
    val_loader = DataLoader(WindowIds(w["val_t"]), batch_size=c["batch_windows"], shuffle=False,
                             collate_fn=collate)

    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    best_val, best_epoch, since_best, history = float("inf"), -1, 0, []
    for epoch in range(c["max_epochs"]):
        train_loss = _run_epoch(model, edge_index, edge_weight, train_loader, scaler, opt)
        val_mae = _run_epoch(model, edge_index, edge_weight, val_loader, scaler, None)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_mae": val_mae})
        if val_mae < best_val:
            best_val, best_epoch, since_best = val_mae, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1
        if since_best >= c["patience_epochs"]:
            break
    model.load_state_dict(best_state)

    run_id = f"gwn-L12-f{int(round(fraction * 100))}-s{seed}"
    run_dir = os.path.join(results, run_id)
    os.makedirs(run_dir, exist_ok=True)

    metrics = {}
    for split in ("val", "test"):
        loader = DataLoader(WindowIds(w[f"{split}_t"]), batch_size=c["batch_windows"], shuffle=False,
                             collate_fn=collate)
        t_arr, yhat_arr = _predict_split(model, edge_index, edge_weight, scaler, loader)
        path = os.path.join(run_dir, f"{split}_preds.npz")
        np.savez(path, t=t_arr, yhat=yhat_arr)
        metrics[split] = score(path, ds, split, data_dir)

    with open(os.path.join(run_dir, "log.jsonl"), "w") as f:
        for row in history:
            f.write(json.dumps(row) + "\n")
    torch.save(best_state, os.path.join(run_dir, "best.pt"))

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    meta = {"git_sha": sha, "run_id": run_id, "fraction": fraction, "seed": seed, "best_epoch": best_epoch,
            "best_val_mae": best_val, "trained_params": sum(p.numel() for p in model.parameters()),
            "frozen_params": 0, "scaler": scaler.state_dict(), "wall_seconds": round(time.time() - t0, 2)}
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
