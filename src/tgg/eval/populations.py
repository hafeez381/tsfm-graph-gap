"""Test-window populations, fixed before any adapter is trained (RD §1.1, guide C4).

onset:  every observed lookback reading >= 0.6 v_i, and some observed target < 0.6 v_i.
steady: every observed lookback and target reading within 10% of the observed lookback mean.
Both need at least one observed lookback reading and at least one observed target.
A window that meets both definitions (a lookback just above 0.6 v_i dipping below it by less than 10%)
is onset only, so the two populations never share a sensor window.
"""
import numpy as np

L, H = 12, 12
ONSET, STEADY, FREEFLOW_PCT = 0.6, 0.10, 85


def free_flow(X: np.ndarray, M: np.ndarray, train_end: int) -> np.ndarray:
    """(N,) 85th percentile of each sensor's observed speeds on training steps [0, train_end) only."""
    x = np.where(M[:train_end] == 1, X[:train_end], np.nan)
    return np.nanpercentile(x, FREEFLOW_PCT, axis=0).astype(np.float32)


def label(X: np.ndarray, M: np.ndarray, t: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(onset, steady), bool arrays of shape (n, N)."""
    lb = t[:, None] - np.arange(L)[None, :]                             # (n, L)
    tg = t[:, None] + np.arange(1, H + 1)[None, :]                      # (n, H)
    xl, ml = X[lb], M[lb].astype(bool)                                  # (n, L, N)
    xt, mt = X[tg], M[tg].astype(bool)
    has = ml.any(1) & mt.any(1)
    thr = ONSET * v[None, :]
    lb_above = np.where(ml, xl >= thr[:, None, :], True).all(1)
    tg_below = np.where(mt, xt < thr[:, None, :], False).any(1)
    onset = has & lb_above & tg_below
    mean = np.where(ml, xl, 0).sum(1) / np.maximum(ml.sum(1), 1)        # (n, N) observed lookback mean
    band = STEADY * mean[:, None, :]
    within = lambda x, m: np.where(m, np.abs(x - mean[:, None, :]) <= band, True).all(1)
    steady = has & within(xl, ml) & within(xt, mt) & ~onset
    return onset, steady
