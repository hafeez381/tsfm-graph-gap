import numpy as np

from tgg.eval import stats

D = 24


def _ds(abs_per_day: np.ndarray, cnt: float = 100.0) -> dict:
    a = np.repeat(abs_per_day[None, :, None], 12, axis=2)             # (1, D, 12)
    return {"days": np.arange(D), "populations": ["all"], "abs": a * cnt, "sq": a * cnt, "cnt": np.full_like(a, cnt)}


def test_identical_models_give_exactly_zero():
    a = _ds(np.random.default_rng(0).uniform(2, 6, D))
    assert stats.paired_diff(a, a) == (0.0, 0.0, 0.0)


def test_constant_offset_is_recovered():
    base = np.random.default_rng(1).uniform(2, 6, D)
    d, lo, hi = stats.paired_diff(_ds(base + 0.3), _ds(base))
    assert np.isclose(d, 0.3) and lo > 0 and np.isclose(lo, 0.3) and np.isclose(hi, 0.3)


def test_gap_closure_half():
    base = np.random.default_rng(2).uniform(3, 5, D)
    g = stats.gap_closure(_ds(base + 1.0), _ds(base + 0.5), _ds(base))
    assert np.isclose(g["G"], 0.5) and np.isclose(g["lo"], 0.5) and g["denom_excludes_0"]


def test_spearman_monotone():
    x = np.arange(50.0)
    rho, lo, hi = stats.spearman_ci(x, x ** 3)
    assert np.isclose(rho, 1.0) and np.isclose(lo, 1.0)
