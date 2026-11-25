import numpy as np
import torch

from scripts import build_cache
from tgg.tsfm import bolt


class _FakeModel(torch.nn.Module):
    def encode(self, context, mask=None):
        B, tokens = context.shape[0], 2
        hidden = torch.ones(B, tokens, bolt.D)
        attention_mask = torch.ones(B, tokens)
        return hidden, (torch.zeros(B, 1), torch.ones(B, 1)), torch.zeros(B, tokens, bolt.D), attention_mask


class _FakePipeline:
    def __init__(self):
        self.model = _FakeModel()

    def predict(self, contexts, prediction_length=12):
        self.model.encode(contexts)
        return torch.zeros(contexts.shape[0], 9, prediction_length) + 42.0


def test_build_split_writes_cache_with_expected_shapes(tmp_path):
    pipeline = _FakePipeline()
    bolt._wrap_encode(pipeline)
    rng = np.random.default_rng(0)
    T, N = 60, 4
    X = (60 + rng.normal(0, 1, (T, N))).astype(np.float32)
    M = np.ones((T, N), dtype=np.uint8)
    t = np.arange(20, 25, dtype=np.int64)                              # 5 windows

    out = build_cache.build_split(pipeline, X, M, t, str(tmp_path / "cache"), batch_series=8, time_first=3)

    arrays = out["arrays"]
    n = len(t)
    assert arrays["h"].shape == (n, N, bolt.D) and arrays["h"].dtype == np.float16
    assert arrays["y0"].shape == (n, N, 12) and np.allclose(arrays["y0"], 42.0)
    assert arrays["obs"].all()                                         # every step observed
    assert np.array_equal(np.asarray(arrays["t"]), t)
    assert out["manifest"]["encoder"] == "bolt-L12"
    assert out["timing"]["windows"] >= 3                                # the fake pipeline is near-instant


def test_lookback_orders_oldest_first_most_recent_last():
    X = np.arange(40.0).reshape(20, 2)
    M = np.ones_like(X, dtype=np.uint8)
    t = np.array([15])
    x, m = build_cache._lookback(X, M, t)
    assert x.shape == (1, 2, build_cache.L)
    assert x[0, 0, -1] == X[15, 0] and x[0, 0, 0] == X[15 - build_cache.L + 1, 0]
    assert m.all()
