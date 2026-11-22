"""Masked metrics over (n, N, H) arrays: only observed targets (m == 1) count."""
import numpy as np


def masked_mae(y: np.ndarray, yhat: np.ndarray, m: np.ndarray) -> float:
    m = m.astype(bool)
    if not m.any():
        raise ValueError("no observed targets")
    return float(np.abs(yhat - y)[m].mean())


def masked_rmse(y: np.ndarray, yhat: np.ndarray, m: np.ndarray) -> float:
    m = m.astype(bool)
    if not m.any():
        raise ValueError("no observed targets")
    return float(np.sqrt(((yhat - y)[m] ** 2).mean()))
