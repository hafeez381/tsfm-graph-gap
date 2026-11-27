from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tgg.tsfm import bolt, finetune

N, T, H, L = 5, 60, 12, 12


class _FakeModel(torch.nn.Module):
    """Mimics ChronosBoltOutput's two fields without importing the real chronos package, so this
    test never puts "chronos" in sys.modules and never trips tgg.adapter.train's encoder-not-loaded
    guard in a later test of the same pytest process."""

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(3))

    def forward(self, context, mask=None, target=None, target_mask=None):
        B = context.shape[0]
        quantile_preds = torch.arange(9, dtype=torch.float32).view(1, 9, 1).expand(B, 9, H) + self.w.sum()
        loss = torch.tensor(1.23) if target is not None else None
        return SimpleNamespace(loss=loss, quantile_preds=quantile_preds)


class _FakePipeline:
    def __init__(self):
        self.model = _FakeModel()


def test_assert_unfrozen_raises_on_a_frozen_parameter():
    pipeline = _FakePipeline()
    finetune.assert_unfrozen(pipeline)
    list(pipeline.model.parameters())[0].requires_grad = False
    with pytest.raises(AssertionError):
        finetune.assert_unfrozen(pipeline)


def test_drop_flat_windows_keeps_only_sufficiently_varying_lookbacks():
    x_raw = np.array([[60.0] * L, [55.0, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66], [0.0] * L])
    m_raw = np.array([[True] * L, [True] * L, [False] * L])
    keep = finetune.drop_flat_windows(x_raw, m_raw)
    assert keep.tolist() == [False, True, False]


def test_contexts_and_targets_shapes_and_nan_at_missing_steps():
    rng = np.random.default_rng(0)
    X = rng.uniform(40, 70, (T, N)).astype(np.float32)
    M = np.ones((T, N), dtype=np.uint8)
    M[10, 2] = 0
    t = np.array([20, 30])
    context, y, my = finetune.contexts_and_targets(X, M, t, L, H)
    assert context.shape == (len(t) * N, L)
    assert y.shape == my.shape == (len(t) * N, H)
    # window t=20, sensor 2's lookback step 10 is masked -> NaN in its context
    row = 0 * N + 2                                                   # t=20 is the first window, sensor 2
    assert np.isnan(context[row]).any()


def test_loss_and_preds_takes_the_median_quantile():
    pipeline = _FakePipeline()
    ctx = torch.zeros(2, L)
    tgt = torch.zeros(2, H)
    tgt_mask = torch.ones(2, H, dtype=torch.bool)
    loss, y0 = finetune.loss_and_preds(pipeline, ctx, tgt, tgt_mask)
    assert loss.item() == pytest.approx(1.23)
    assert y0.shape == (2, H)
    assert torch.allclose(y0, torch.full((2, H), float(bolt.MEDIAN_IDX)))
