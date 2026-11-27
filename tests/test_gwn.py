import numpy as np
import torch

from tgg.baselines import gwn

N, T, L, H = 10, 400, 12, 12


def _graph():
    rng = np.random.default_rng(0)
    upper = np.triu(rng.random((N, N)) < 0.3, 1)
    return (upper | upper.T).astype(np.float32)


def test_scaler_fits_on_training_steps_only():
    X_filled = np.arange(T, dtype=np.float32)[:, None].repeat(N, 1)
    train_end = 100
    scaler = gwn.Scaler.fit(X_filled, train_end)
    assert abs(scaler.mean - X_filled[:train_end].mean()) < 1e-3
    assert abs(scaler.std - X_filled[:train_end].std()) < 1e-3
    x = np.array([scaler.mean], dtype=np.float32)
    assert round(float(scaler.transform(x)[0]), 6) == 0.0
    assert round(float(scaler.inverse(scaler.transform(x))[0]), 4) == round(scaler.mean, 4)


def test_build_has_about_300k_params_and_is_trainable():
    model, edge_index, edge_weight = gwn.build(N, _graph())
    n = sum(p.numel() for p in model.parameters())
    assert 250_000 < n < 350_000, n
    assert all(p.requires_grad for p in model.parameters())


def test_predict_output_shape_is_B_N_H():
    model, edge_index, edge_weight = gwn.build(N, _graph())
    scaler = gwn.Scaler(mean=60.0, std=5.0)
    x = torch.randn(4, L, N, 2)
    yhat = gwn.predict(model, edge_index, edge_weight, x, scaler)
    assert yhat.shape == (4, N, H)


def test_predict_is_denormalised_to_mph_scale():
    """A model whose readout is forced to 0 must predict exactly the scaler's mean."""
    model, edge_index, edge_weight = gwn.build(N, _graph())
    for p in model.parameters():
        p.detach().zero_()
    scaler = gwn.Scaler(mean=61.5, std=5.2)
    x = torch.randn(2, L, N, 2)
    yhat = gwn.predict(model, edge_index, edge_weight, x, scaler)
    assert torch.allclose(yhat, torch.full_like(yhat, 61.5), atol=1e-4)


def test_make_input_is_zscored_and_time_ordered():
    rng = np.random.default_rng(1)
    X_filled = rng.normal(60, 5, (T, N)).astype(np.float32)
    tod = (np.arange(T) % 288 / 288.0).astype(np.float32)
    scaler = gwn.Scaler.fit(X_filled, 200)
    t = np.array([250, 300])
    x = gwn.make_input(X_filled, tod, t, scaler)
    assert x.shape == (2, L, N, 2)
    # last lookback step (index -1) is t itself
    expected = scaler.transform(X_filled[t])
    assert np.allclose(x[:, -1, :, 0], expected, atol=1e-5)
    assert np.allclose(x[:, -1, :, 1], tod[t][:, None], atol=1e-5)


def test_lookback_ids_are_oldest_first():
    t = np.array([50])
    lb = gwn.lookback_ids(t, Lw=L)
    assert lb.shape == (1, L)
    assert lb[0, -1] == 50
    assert lb[0, 0] == 50 - (L - 1)
    assert np.array_equal(lb[0], np.arange(50 - L + 1, 51))
