import numpy as np
import pytest
import torch

from tgg.tsfm import timesfm as tfm


class _FakeLayer(torch.nn.Module):
    def forward(self, x):
        return x, None                                                  # (out, cache), guide C8


class _FakeModelInner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.zeros(2))
        self.stacked_xf = torch.nn.ModuleList([_FakeLayer()])


class _FakeTimesFM:
    """Mimics forecast()'s batch-size padding, so run()'s trim-to-len logic is exercised for real."""

    def __init__(self, global_batch_size=3, num_patches=2, width=tfm.D_RAW):
        self.model = _FakeModelInner()
        self.global_batch_size = global_batch_size
        self.num_patches = num_patches
        self.width = width

    def forecast(self, horizon, inputs):
        n = len(inputs)
        gbs = self.global_batch_size
        total = n + (-n) % gbs
        for start in range(0, total, gbs):
            x = torch.arange(start, start + gbs).float().view(gbs, 1, 1).repeat(1, self.num_patches, self.width)
            self.model.stacked_xf[-1](x)                                 # fires the forward hook
        return np.zeros((n, horizon)), np.zeros((n, horizon, 10))


def test_run_trims_padding_and_keeps_last_patch_token():
    model = _FakeTimesFM(global_batch_size=3, num_patches=2)
    tfm._wrap_last_layer(model)
    lookbacks = [np.zeros(12)] * 7                                       # not a multiple of 3
    y0, h_raw = tfm.run(model, lookbacks, horizon=12)
    assert y0.shape == (7, 12)
    assert h_raw.shape == (7, tfm.D_RAW)
    assert np.allclose(h_raw[:, 0], np.arange(7))                        # no misalignment from padding


def test_assert_frozen():
    model = _FakeTimesFM()
    with pytest.raises(AssertionError):
        tfm.assert_frozen(model)
    for p in model.model.parameters():
        p.requires_grad = False
    tfm.assert_frozen(model)


def test_fit_pca_explains_variance_and_apply_matches_shape():
    rng = np.random.default_rng(0)
    base = rng.normal(size=(2000, 20))
    projection = rng.normal(size=(20, 50))
    data = (base @ projection).astype(np.float32)                        # rank <= 20
    pca = tfm.fit_pca(data, n_components=20)
    reduced = tfm.apply_pca(pca, data[:5])
    assert reduced.shape == (5, 20)


def test_fit_pca_raises_when_variance_insufficient():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(500, 100)).astype(np.float32)                # isotropic noise
    with pytest.raises(AssertionError):
        tfm.fit_pca(data, n_components=2)
