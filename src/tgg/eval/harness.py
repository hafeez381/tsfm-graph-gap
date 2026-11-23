"""The one scorer: a predictions file in, metrics.json and day_sums.npz out (guide C2).

Scored set Omega (eq. 2): observed targets of sensor windows whose 12-step lookback holds at least one
observed reading. A window's day is the calendar day of its first target step t + 1 (Decision 15).
"""
import json
import os

import numpy as np

H, L_SCORE = 12, 12
REPORT_H = (3, 6, 12)


def assert_common_targets(t: np.ndarray, expected: np.ndarray) -> None:
    """Raise unless the predictions cover exactly the split's window ids, in order (Decision 3)."""
    if t.shape != expected.shape or not np.array_equal(t, expected):
        raise AssertionError(f"window ids differ from the split: {len(t)} given, {len(expected)} expected")


def targets(X: np.ndarray, M: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """y, m of shape (n, N, H) from raw X and M (never X_filled)."""
    idx = t[:, None] + np.arange(1, H + 1)[None, :]
    return X[idx].transpose(0, 2, 1), M[idx].transpose(0, 2, 1).astype(bool)


def observed_lookback(M: np.ndarray, t: np.ndarray, L: int = L_SCORE) -> np.ndarray:
    """(n, N) bool: at least one observed reading in steps t-L+1 .. t."""
    return M[t[:, None] - np.arange(L)[None, :]].any(axis=1)


def _populations(path: str | None, split: str, n: int, N: int) -> dict[str, np.ndarray]:
    """Population masks of shape (n, N). P_all always; the rest from data/<ds>_populations.npz (C4)."""
    pops = {"all": np.ones((n, N), dtype=bool)}
    if path and os.path.exists(path):
        p = np.load(path)
        for name in ("onset", "steady"):
            key = f"{name}_{split}"
            if key in p:
                pops[name] = p[key].astype(bool)
        for name in ("graph_hi", "graph_lo"):
            if name in p:
                pops[name] = np.broadcast_to(p[name].astype(bool)[None, :], (n, N))
    for k, v in pops.items():
        assert v.shape == (n, N), (k, v.shape)
    return pops


def score(preds_path: str, ds: str = "metr_la", split: str = "test", data_dir: str = "data") -> dict:
    preds = np.load(preds_path)
    t, yhat = preds["t"].astype(np.int64), preds["yhat"]
    w = np.load(os.path.join(data_dir, f"{ds}_windows.npz"))
    assert_common_targets(t, w[f"{split}_t"].astype(np.int64))
    d = np.load(os.path.join(data_dir, f"{ds}.npz"))
    X, M, ts = d["X"], d["M"], d["ts"]
    y, m = targets(X, M, t)
    if yhat.shape != y.shape or not np.isfinite(yhat).all():
        raise AssertionError(f"yhat must be finite with shape {y.shape}, got {yhat.shape}")
    keep = observed_lookback(M, t)                                    # (n, N)
    omega = m & keep[..., None]                                       # (n, N, H)

    err = (yhat - y).astype(np.float64)
    ae, se = np.abs(err), err ** 2
    day = (ts[t + 1] // 86400).astype(np.int64)                       # UTC day of the first target step
    days, day_idx = np.unique(day, return_inverse=True)

    pops = _populations(os.path.join(data_dir, f"{ds}_populations.npz"), split, *keep.shape)
    sums = {k: np.zeros((len(pops), len(days), H)) for k in ("abs", "sq", "cnt")}
    metrics = {"split": split, "n_windows": int(len(t)), "n_sensor_windows": int(keep.size),
               "n_excluded_no_lookback": int((~keep).sum()),
               "share_excluded_no_lookback": float((~keep).mean()), "populations": {}}
    for p, (name, pm) in enumerate(pops.items()):
        sel = omega & pm[..., None]
        for h in range(H):
            s = sel[:, :, h]
            for k, v in (("abs", ae[:, :, h]), ("sq", se[:, :, h]), ("cnt", s)):
                sums[k][p, :, h] = np.bincount(day_idx, weights=np.where(s, v, 0).sum(1), minlength=len(days))
        cnt = sel.sum((0, 1))                                         # (H,) observed targets per horizon
        row = {"n_targets": int(cnt.sum())}
        if cnt.sum() == 0:
            metrics["populations"][name] = row
            continue
        for h in REPORT_H:
            row[f"mae_h{h}"] = float(ae[:, :, h - 1][sel[:, :, h - 1]].mean())
            row[f"rmse_h{h}"] = float(np.sqrt(se[:, :, h - 1][sel[:, :, h - 1]].mean()))
        row["mae_avg"] = float(ae[sel].mean())
        row["rmse_avg"] = float(np.sqrt(se[sel].mean()))
        metrics["populations"][name] = row

    out = os.path.dirname(os.path.abspath(preds_path))
    prefix = "" if split == "test" else f"{split}_"
    with open(os.path.join(out, f"{prefix}metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez(os.path.join(out, f"{prefix}day_sums.npz"), days=days, populations=np.array(list(pops)), **sums)
    return metrics
