"""Write and score the last-value and historical-average floors on validation and test.

Usage: python -m scripts.score_floors [--ds metr_la] [--data data] [--results results]
"""
import argparse
import json
import os

import subprocess
import time

import numpy as np

from tgg.baselines import naive
from tgg.eval.harness import score


def run(ds: str, data: str, results: str) -> dict:
    d = np.load(os.path.join(data, f"{ds}.npz"))
    w = np.load(os.path.join(data, f"{ds}_windows.npz"))
    X, M, ts = d["X"], d["M"], d["ts"]
    table = naive.historical_average(X, M, ts, int(w["train_end"]))
    out = {}
    for name, fn in (("last-value", lambda t: naive.last_value(X, M, ts, t, table)),
                     ("hist-avg", lambda t: naive.predict_ha(table, ts, t))):
        run_dir = os.path.join(results, f"{name}-{ds}")
        os.makedirs(run_dir, exist_ok=True)
        for split in ("val", "test"):
            t = w[f"{split}_t"]
            path = os.path.join(run_dir, f"{split}_preds.npz")
            np.savez(path, t=t, yhat=fn(t))
            out[f"{name}/{split}"] = score(path, ds, split, data)["populations"]["all"]
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        meta = {"git_sha": sha, "model": name, "dataset": ds, "trained_params": 0, "frozen_params": 0,
                "fit": "training steps only" if name == "hist-avg" else "none (lookback only, HA fallback)",
                "created": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        with open(os.path.join(run_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="metr_la")
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results")
    print(json.dumps(run(**vars(ap.parse_args())), indent=2))
