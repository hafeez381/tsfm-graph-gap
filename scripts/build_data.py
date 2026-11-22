"""Write the data, windows and graph interface files for one dataset, plus a stats file.

Usage: python -m scripts.build_data --dataset metr_la [--root /tmp/tsl_data/metr_la] [--out data]
"""
import argparse
import json
import os

import numpy as np

from tgg.data import graph, loaders, windows


def run(dataset: str, root: str, out: str) -> dict:
    d, ds = loaders.load(dataset, root)
    A, nbr = graph.build(ds)
    T, N = d["X"].shape
    w = windows.make(T)
    os.makedirs(out, exist_ok=True)
    np.savez(os.path.join(out, f"{dataset}.npz"), **d)
    np.savez(os.path.join(out, f"{dataset}_windows.npz"), **w)
    np.savez(os.path.join(out, f"{dataset}_graph.npz"), A=A, nbr=nbr)

    steps_per_day = 288
    n_steps = {"train": w["train_end"], "val": w["val_end"] - w["train_end"], "test": T - w["val_end"]}
    f5 = windows.fraction_block(w["train_t"], 0.05)
    stats = {
        "T": T, "N": N,
        "start": str(ds.index[0]), "end": str(ds.index[-1]),
        "edges_offdiag": int((A > 0).sum()),
        "edges_symmetric_pairs": int(((A > 0) & (A.T > 0)).sum() // 2),
        "zero_share": float(1 - d["M"].mean()),
        "leading_zeros_unfilled": int((d["X_filled"] == 0).sum()),
        "split_steps": n_steps,
        "split_days": {k: round(v / steps_per_day, 2) for k, v in n_steps.items()},
        "split_windows": {k: len(w[f"{k}_t"]) for k in ("train", "val", "test")},
        "train_end": w["train_end"], "val_end": w["val_end"],
        "nbr_size_median": float(np.median(nbr.sum(1))),
        "nbr_size_min": int(nbr.sum(1).min()), "nbr_size_max": int(nbr.sum(1).max()),
        "isolated_sensors": np.flatnonzero(nbr.sum(1) == 1).tolist(),
        "fraction_windows": {str(f): len(windows.fraction_block(w["train_t"], f)) for f in (0.05, 0.25, 1.0)},
        "fraction_5pct_days": round(len(f5) / steps_per_day, 2),
    }
    with open(os.path.join(out, f"{dataset}_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="metr_la", choices=sorted(loaders.DATASETS))
    ap.add_argument("--root", default="/tmp/tsl_data/metr_la")
    ap.add_argument("--out", default="data")
    print(json.dumps(run(**vars(ap.parse_args())), indent=2))
