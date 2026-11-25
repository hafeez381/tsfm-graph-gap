import os

import numpy as np

from scripts import probe
from tgg.data import windows
from tgg.tsfm import cache

T, N, D = 2000, 5, 8


def _build_fake_cache(tmp_path, X, M, w, sd_value=1.0):
    """A cache directory whose h[...,0] equals the exact trend target (so ridge predicts it
    perfectly) and whose other dims are noise uncorrelated with speed level."""
    rng = np.random.default_rng(0)
    cache_dir = str(tmp_path / "bolt-L12")
    for split in ("train", "val"):
        t = w[f"{split}_t"]
        n = len(t)
        arrays = cache.create(os.path.join(cache_dir, split), n, N, D)
        arrays["t"][:] = t
        arrays["obs"][:] = True
        arrays["sd"][:] = sd_value
        y_trend, _ = probe.trend_target(X, M, t, np.full((n, N), sd_value))
        h = rng.normal(0, 1, (n, N, D)).astype(np.float32)
        h[:, :, 0] = y_trend                                          # perfectly informative dimension
        arrays["h"][:] = h.astype(np.float16)
        for a in arrays.values():
            a.flush()
    return cache_dir


def test_probe_finds_trend_in_engineered_dimension_not_level(tmp_path):
    rng = np.random.default_rng(1)
    steps = np.arange(T)
    X = (60 + 8 * np.sin(2 * np.pi * steps / 288)[:, None] + rng.normal(0, 0.5, (T, N))).astype(np.float32)
    M = np.ones((T, N), dtype=np.uint8)
    w = windows.make(T, L_long=288, H=12)

    cache_dir = _build_fake_cache(tmp_path, X, M, w)
    out = probe.run_probe(cache_dir, X, M, h_field="h", n_train=500, n_val=200)

    assert out["trend_r2"] > 0.95                                      # h[...,0] IS the target exactly
    assert out["level_r2"] < out["trend_r2"]                            # noise dims carry no level info


def test_trend_target_requires_both_ends_observed():
    X = np.array([[10.0], [20.0], [30.0]] * 10)                        # (30, 1)
    M = np.ones_like(X, dtype=np.uint8)
    M[5, 0] = 0                                                        # one missing reading
    t = np.array([16])                                                 # t - 11 = 5, the missing step
    y, both_obs = probe.trend_target(X, M, t, sd=np.ones((1, 1)))
    assert not both_obs[0, 0] and np.isnan(y[0, 0])
