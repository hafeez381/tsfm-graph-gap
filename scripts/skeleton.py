"""Walking skeleton: synthetic data -> windows -> fake cache -> zero-initialised stub adapter -> predictions -> score.

Every stage is a stand-in for the real one, so the interfaces between workstreams are exercised on day one.
Usage: python -m scripts.skeleton [--out DIR] [--steps 50]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from tgg.data import windows
from tgg.eval.metrics import masked_mae
from tgg.seed import GLOBAL_SEED, seed_everything

T, N, H, L, D = 600, 10, 12, 12, 512


def make_data(rng: np.random.Generator) -> dict:
    """Synthetic speeds in mph with a daily cycle, 5% zeros as missing, and a random symmetric neighbour mask."""
    steps = np.arange(T)
    X = 60 + 8 * np.sin(2 * np.pi * steps / 288)[:, None] + rng.normal(0, 2, (T, N))
    M = (rng.random((T, N)) >= 0.05).astype(np.uint8)
    X = (X * M).astype(np.float32)
    X_filled = X.copy()
    for t in range(1, T):  # forward fill, as tsl does
        X_filled[t] = np.where(M[t] == 1, X[t], X_filled[t - 1])
    upper = np.triu(rng.random((N, N)) < 0.3, 1)
    nbr = upper | upper.T | np.eye(N, dtype=bool)
    tod = (steps % 288) * 5 / 1440
    return {"X": X, "M": M, "X_filled": X_filled, "tod": tod.astype(np.float32),
            "ts": (1_330_000_000 + 300 * steps).astype(np.int64), "nbr": nbr}


def make_cache(d: dict, t: np.ndarray, rng: np.random.Generator) -> dict:
    """Fake cache for windows t: random embeddings, y0 = last filled value repeated, s = mean |observed lookback|."""
    lb = np.stack([np.arange(i - L + 1, i + 1) for i in t])          # (n, L) lookback step indices
    xl, ml = d["X"][lb], d["M"][lb].astype(bool)                       # (n, L, N)
    obs = ml.any(axis=1)                                               # (n, N)
    s = np.where(obs, np.abs(xl).sum(1) / np.maximum(ml.sum(1), 1), 1.0).astype(np.float32)
    y0 = np.repeat(d["X_filled"][t][:, :, None], H, axis=2).astype(np.float32)
    h = rng.normal(0, 1, (len(t), N, D)).astype(np.float16)
    return {"t": t, "h": h, "y0": y0, "s": s, "obs": obs}


def targets(d: dict, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Targets from raw X (never X_filled) and their mask, shape (n, N, H)."""
    idx = t[:, None] + np.arange(1, H + 1)[None, :]                    # (n, H)
    return d["X"][idx].transpose(0, 2, 1), d["M"][idx].transpose(0, 2, 1)


class StubAdapter(torch.nn.Module):
    """yhat = y0 + s * (W h + b) with W, b zero-initialised, so step 0 returns the zero-shot forecast exactly."""

    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(D, H)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, h, y0, s):
        return y0 + s[..., None] * self.head(h.float())


def run(out: str, steps: int = 50) -> dict:
    t0 = time.time()
    seed_everything(GLOBAL_SEED)
    rng = np.random.default_rng(GLOBAL_SEED)
    d = make_data(rng)
    os.makedirs("data", exist_ok=True)
    np.savez("data/stub.npz", **d)
    w = windows.make(T)
    tr, te = make_cache(d, w["train_t"], rng), make_cache(d, w["test_t"], rng)
    y_tr, m_tr = targets(d, w["train_t"])
    y_te, m_te = targets(d, w["test_t"])
    keep_te = m_te * te["obs"][..., None]                              # drop sensor windows with no observed lookback

    to = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32))
    model = StubAdapter()
    with torch.no_grad():
        step0 = model(to(te["h"]), to(te["y0"]), to(te["s"])).numpy()
    last_value_mae = masked_mae(y_te, te["y0"], keep_te)
    step0_mae = masked_mae(y_te, step0, keep_te)
    assert step0_mae == last_value_mae, (step0_mae, last_value_mae)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    keep_tr = to(m_tr * tr["obs"][..., None])
    for _ in range(steps):
        yhat = model(to(tr["h"]), to(tr["y0"]), to(tr["s"]))
        loss = (torch.abs(yhat - to(y_tr)) * keep_tr).sum() / keep_tr.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        yhat_te = model(to(te["h"]), to(te["y0"]), to(te["s"])).numpy()
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, "test_preds.npz"), t=w["test_t"], yhat=yhat_te.astype(np.float32))
    preds = np.load(os.path.join(out, "test_preds.npz"))               # score what was written, not what is in memory
    assert np.array_equal(preds["t"], w["test_t"])
    metrics = {"last_value_mae": last_value_mae, "step0_mae": step0_mae,
               "trained_mae": masked_mae(y_te, preds["yhat"], keep_te), "train_loss_final": loss.item(),
               "n_train": len(w["train_t"]), "n_test": len(w["test_t"]), "steps": steps,
               "device": "cuda" if torch.cuda.is_available() else "cpu", "seconds": round(time.time() - t0, 2)}
    with open(os.path.join(out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.environ.get("OUT_DIR", "."), "results/skeleton"))
    ap.add_argument("--steps", type=int, default=50)
    print(json.dumps(run(**vars(ap.parse_args())), indent=2))
