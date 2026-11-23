"""Graph dependence Gamma_i (eq. 7): how much a ridge regression gains from neighbours' lookbacks.

Per sensor i, two ridge regressions from 12-step lookbacks of X_filled to the 12 targets:
self (x_i only) and nbr (x_i plus x_j, j in N(i) minus i). Fitted on training windows whose 12 targets
are all observed; the penalty is chosen from LAMBDAS by masked validation MAE, which is also the score.
Gamma_i = (M_self - M_nbr) / M_self. A sensor with no neighbour gets Gamma_i = 0.
"""
import numpy as np

L, H = 12, 12
LAMBDAS = (0.1, 1.0, 10.0, 100.0)


def _features(Xf: np.ndarray, t: np.ndarray, cols: np.ndarray) -> np.ndarray:
    lb = t[:, None] - np.arange(L)[None, :]                             # (n, L)
    return Xf[lb][:, :, cols].reshape(len(t), -1)                       # (n, L * len(cols))


def _ridge_val_mae(F_tr, Y_tr, F_va, Y_va, m_va) -> float:
    mu, sd = F_tr.mean(0), F_tr.std(0) + 1e-6
    A, B = (F_tr - mu) / sd, (F_va - mu) / sd
    ym = Y_tr.mean(0)
    G, R = A.T @ A, A.T @ (Y_tr - ym)
    best = np.inf
    for lam in LAMBDAS:
        W = np.linalg.solve(G + lam * len(A) * np.eye(G.shape[0]), R)   # penalty scaled per sample
        err = np.abs(B @ W + ym - Y_va)[m_va]
        best = min(best, float(err.mean()))
    return best


def fit(X: np.ndarray, X_filled: np.ndarray, M: np.ndarray, nbr: np.ndarray, train_t: np.ndarray,
        val_t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(gamma, mae_self, mae_nbr), each (N,)."""
    N = X.shape[1]
    tg = lambda t: t[:, None] + np.arange(1, H + 1)[None, :]
    Y_tr, M_tr = X[tg(train_t)], M[tg(train_t)].astype(bool)             # (n, H, N)
    Y_va, M_va = X[tg(val_t)], M[tg(val_t)].astype(bool)
    obs_va = M[val_t[:, None] - np.arange(L)[None, :]].any(1)           # (n, N): scored like Omega
    gamma, ms, mn = np.zeros(N), np.zeros(N), np.zeros(N)
    for i in range(N):
        rows = M_tr[:, :, i].all(1)
        mask = M_va[:, :, i] & obs_va[:, i][:, None]
        others = np.flatnonzero(nbr[i] & (np.arange(N) != i))
        ytr, yva = Y_tr[rows][:, :, i], Y_va[:, :, i]
        ms[i] = _ridge_val_mae(_features(X_filled, train_t[rows], np.array([i])), ytr,
                               _features(X_filled, val_t, np.array([i])), yva, mask)
        if len(others) == 0:
            mn[i] = ms[i]
            continue
        cols = np.r_[i, others]
        mn[i] = _ridge_val_mae(_features(X_filled, train_t[rows], cols), ytr,
                               _features(X_filled, val_t, cols), yva, mask)
        gamma[i] = (ms[i] - mn[i]) / ms[i]
    return gamma, ms, mn
