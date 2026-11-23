"""Write data/<ds>_populations.npz and results/populations-<ds>/summary.json (guide C4).

Usage: python -m scripts.build_populations [--ds metr_la] [--data data] [--results results]
"""
import argparse
import hashlib
import json
import os

import numpy as np

from tgg.eval import gamma, populations


def run(ds: str, data: str, results: str) -> dict:
    d = np.load(os.path.join(data, f"{ds}.npz"))
    w = np.load(os.path.join(data, f"{ds}_windows.npz"))
    nbr = np.load(os.path.join(data, f"{ds}_graph.npz"))["nbr"]
    X, M, Xf, train_end = d["X"], d["M"], d["X_filled"], int(w["train_end"])
    v = populations.free_flow(X, M, train_end)
    out = {"v": v}
    summary = {"dataset": ds, "free_flow_mph": {"median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}}
    for split in ("val", "test"):
        t = w[f"{split}_t"]
        onset, steady = populations.label(X, M, t, v)
        obs = M[t[:, None] - np.arange(12)[None, :]].any(1)            # sensor windows that are scored
        out[f"onset_{split}"], out[f"steady_{split}"] = onset, steady
        summary[split] = {"sensor_windows_scored": int(obs.sum()), "onset": int(onset.sum()), "steady": int(steady.sum()),
                          "onset_share": float(onset.sum() / obs.sum()), "steady_share": float(steady.sum() / obs.sum())}
    g, ms, mn = gamma.fit(X, Xf, M, nbr, w["train_t"], w["val_t"])
    order = np.argsort(g, kind="stable")
    third = len(g) // 3
    hi, lo = np.zeros(len(g), bool), np.zeros(len(g), bool)
    hi[order[-third:]], lo[order[:third]] = True, True
    out.update(gamma=g.astype(np.float32), mae_self=ms.astype(np.float32), mae_nbr=mn.astype(np.float32), graph_hi=hi, graph_lo=lo)
    summary["gamma"] = {"median": float(np.median(g)), "mean": float(g.mean()), "min": float(g.min()), "max": float(g.max()),
                        "share_positive": float((g > 0).mean()), "hi_threshold": float(g[order[-third]]),
                        "lo_threshold": float(g[order[third - 1]]), "n_hi": int(hi.sum()), "n_lo": int(lo.sum())}
    path = os.path.join(data, f"{ds}_populations.npz")
    np.savez(path, **out)
    summary["sha256"] = hashlib.sha256(open(path, "rb").read()).hexdigest()
    os.makedirs(os.path.join(results, f"populations-{ds}"), exist_ok=True)
    with open(os.path.join(results, f"populations-{ds}", "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="metr_la")
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results")
    print(json.dumps(run(**vars(ap.parse_args())), indent=2))
