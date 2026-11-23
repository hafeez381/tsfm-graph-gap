"""Paired day bootstrap (Decision 15, eq. 9), gap-closure intervals and Spearman intervals.

Day sums come from `score()`: arrays abs, sq, cnt of shape (populations, days, 12) with `days`, `populations`.
Every resample draws test days with replacement once and applies the same draw to every model compared.
"""
import numpy as np
from scipy.stats import spearmanr

N_BOOT, SEED, CI = 1000, 651, 0.95


def load(path: str) -> dict:
    z = np.load(path)
    return {"days": z["days"], "populations": list(z["populations"]), "abs": z["abs"], "sq": z["sq"], "cnt": z["cnt"]}


def _select(ds: dict, pop: str, h) -> tuple[np.ndarray, np.ndarray]:
    """Per-day absolute-error sums and counts (days,) for one population at horizon h (1..12) or 'avg'."""
    p = ds["populations"].index(pop)
    a, c = ds["abs"][p], ds["cnt"][p]
    return (a.sum(1), c.sum(1)) if h == "avg" else (a[:, h - 1], c[:, h - 1])


def _draws(n_days: int, n: int, seed: int) -> np.ndarray:
    """(n, n_days) multiplicity of each day in each resample."""
    idx = np.random.default_rng(seed).integers(0, n_days, (n, n_days))
    return np.stack([np.bincount(r, minlength=n_days) for r in idx]).astype(np.float64)


def _mae_draws(ds: dict, pop: str, h, w: np.ndarray) -> tuple[float, np.ndarray]:
    a, c = _select(ds, pop, h)
    return a.sum() / c.sum(), (w @ a) / (w @ c)


def _check_paired(*dss: dict) -> None:
    for d in dss[1:]:
        if not np.array_equal(d["days"], dss[0]["days"]) or not np.array_equal(d["cnt"], dss[0]["cnt"]):
            raise AssertionError("models were not scored on the same days and targets")


def _interval(x: np.ndarray, ci: float = CI) -> tuple[float, float]:
    lo, hi = np.percentile(x, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return float(lo), float(hi)


def paired_diff(a: dict, b: dict, pop: str = "all", h=12, n: int = N_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    """MAE_a - MAE_b on all test days, and its 95% paired day-bootstrap interval."""
    _check_paired(a, b)
    w = _draws(len(a["days"]), n, seed)
    ma, da = _mae_draws(a, pop, h, w)
    mb, db = _mae_draws(b, pop, h, w)
    return (float(ma - mb), *_interval(da - db))


def gap_closure(zs: dict, ad: dict, st: dict, pop: str = "all", h=12, n: int = N_BOOT, seed: int = SEED) -> dict:
    """G = (M_zs - M_ad) / (M_zs - M_st) (eq. 6), with numerator and denominator recomputed on every draw (A7).

    The G interval is only meaningful when the denominator's interval excludes 0; `denom_excludes_0` says so.
    """
    _check_paired(zs, ad, st)
    w = _draws(len(zs["days"]), n, seed)
    (mz, dz), (ma, da), (ms, dst) = (_mae_draws(x, pop, h, w) for x in (zs, ad, st))
    denom = dz - dst
    lo, hi = _interval(denom)
    g_lo, g_hi = _interval((dz - da) / denom)
    return {"G": float((mz - ma) / (mz - ms)), "lo": g_lo, "hi": g_hi,
            "denom": float(mz - ms), "denom_lo": lo, "denom_hi": hi, "denom_excludes_0": bool(lo > 0 or hi < 0)}


def spearman_ci(x: np.ndarray, y: np.ndarray, n: int = N_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    """Spearman rho over sensors with finite x and y, and a 95% interval from resampling sensors."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[ok], np.asarray(y)[ok]
    rng = np.random.default_rng(seed)
    boot = [spearmanr(x[i], y[i]).statistic for i in rng.integers(0, len(x), (n, len(x)))]
    return (float(spearmanr(x, y).statistic), *_interval(np.asarray(boot)))
