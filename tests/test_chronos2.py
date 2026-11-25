import numpy as np
import pytest
import torch

from tgg.tsfm import chronos2


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(2))


class _FakePipeline:
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def __init__(self):
        self.model = _FakeModel()

    def predict(self, batch, prediction_length=12):
        n, N, _ = batch.shape
        preds = []
        for i in range(n):
            p = torch.zeros(N, len(self.quantiles), prediction_length)
            p[:, 4, :] = batch[i, :, -1:].expand(N, prediction_length)   # median = last lookback value
            preds.append(p)
        return preds


def test_run_picks_median_quantile_and_preserves_window_order():
    pipeline = _FakePipeline()
    rng = np.random.default_rng(0)
    windows = rng.normal(60, 5, size=(5, 3, 12)).astype(np.float32)      # 5 windows, 3 sensors, L=12
    out = chronos2.run(pipeline, windows, batch_size=2)                  # forces multiple predict() calls
    assert out.shape == (5, 3, 12)
    expected = np.repeat(windows[:, :, -1:], 12, axis=-1)
    assert np.allclose(out, expected)


def test_assert_frozen():
    pipeline = _FakePipeline()
    with pytest.raises(AssertionError):
        chronos2.assert_frozen(pipeline)
    for p in pipeline.model.parameters():
        p.requires_grad = False
    chronos2.assert_frozen(pipeline)
