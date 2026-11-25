import pytest
import torch

from tgg.adapter.model import Adapter, MaskedGAT, combine_mask, masks

N, D, H, B = 10, 512, 12, 3


@pytest.fixture
def nbr():
    rng = torch.Generator().manual_seed(0)
    upper = torch.triu(torch.rand(N, N, generator=rng) < 0.3, diagonal=1)
    return upper | upper.T | torch.eye(N, dtype=torch.bool)


@pytest.fixture
def batch():
    rng = torch.Generator().manual_seed(1)
    h = torch.randn(B, N, D, generator=rng)
    y0 = torch.randn(B, N, H, generator=rng)
    s = torch.rand(B, N, generator=rng) + 0.5
    obs = torch.ones(B, N, dtype=torch.bool)
    return h, y0, s, obs


def test_param_count_is_37836_for_every_variant(nbr):
    for variant in ("mlp", "gat", "full"):
        adapter = Adapter(d=D)
        n_params = sum(p.numel() for p in adapter.parameters())
        assert n_params == 37836, (variant, n_params)


def test_yhat_equals_y0_at_init(nbr, batch):
    h, y0, s, obs = batch
    m = combine_mask(masks(nbr)["gat"], obs)
    adapter = Adapter(d=D)
    yhat, _ = adapter(h, y0, s, m)
    assert torch.equal(yhat, y0)                                     # A3: W_g, b_g start at zero


def test_unobserved_sender_does_not_affect_other_receivers(nbr, batch):
    h, y0, s, obs = batch
    obs = obs.clone()
    obs[:, 3] = False
    m = combine_mask(masks(nbr)["full"], obs)
    adapter = Adapter(d=D)
    with torch.no_grad():
        adapter.Wg.weight.fill_(0.01)                                 # break the zero-init so u matters
    yhat1, _ = adapter(h, y0, s, m)
    h2 = h.clone()
    h2[:, 3, :] = torch.randn(B, D)
    yhat2, _ = adapter(h2, y0, s, m)
    others = [i for i in range(N) if i != 3]
    assert torch.equal(yhat1[:, others, :], yhat2[:, others, :])


def test_mlp_variant_is_fully_independent_across_sensors(nbr, batch):
    h, y0, s, obs = batch
    m = combine_mask(masks(nbr)["mlp"], obs)
    adapter = Adapter(d=D)
    with torch.no_grad():
        adapter.Wg.weight.fill_(0.01)
    yhat1, _ = adapter(h, y0, s, m)
    h2 = h.clone()
    h2[:, 5, :] = torch.randn(B, D)
    yhat2, _ = adapter(h2, y0, s, m)
    others = [i for i in range(N) if i != 5]
    assert torch.equal(yhat1[:, others, :], yhat2[:, others, :])


def test_gat_variant_only_reads_neighbours(nbr, batch):
    h, y0, s, obs = batch
    m = combine_mask(masks(nbr)["gat"], obs)
    adapter = Adapter(d=D)
    with torch.no_grad():
        adapter.Wg.weight.fill_(0.01)
    yhat1, _ = adapter(h, y0, s, m)
    h2 = h.clone()
    i = 0
    outside = [j for j in range(N) if j != i and not nbr[i, j]]
    assert outside, "fixture's graph must leave at least one non-neighbour of sensor 0"
    h2[:, outside[0], :] = torch.randn(B, D)
    yhat2, _ = adapter(h2, y0, s, m)
    assert torch.equal(yhat1[:, i, :], yhat2[:, i, :])


def test_alpha_rows_sum_to_one_and_zero_outside_mask(nbr, batch):
    h, y0, s, obs = batch
    m = combine_mask(masks(nbr)["gat"], obs)
    adapter = Adapter(d=D)
    _, alpha = adapter(h, y0, s, m)
    assert torch.allclose(alpha.sum(-1), torch.ones_like(alpha.sum(-1)), atol=1e-6)
    assert (alpha[~m.unsqueeze(1).expand_as(alpha)] == 0).all()


def test_no_allowed_sender_raises():
    bad_mask = torch.zeros(1, N, N, dtype=torch.bool)                 # diagonal missing
    gat = MaskedGAT(m=64, heads=4)
    with pytest.raises(AssertionError):
        gat(torch.randn(1, N, 64), bad_mask)


def test_permuting_sensors_permutes_the_output(nbr, batch):
    h, y0, s, obs = batch
    adapter = Adapter(d=D)
    with torch.no_grad():
        adapter.Wg.weight.fill_(0.01)
    perm = torch.randperm(N)

    m = combine_mask(masks(nbr)["gat"], obs)
    yhat, _ = adapter(h, y0, s, m)

    nbr_p = nbr[perm][:, perm]
    m_p = combine_mask(masks(nbr_p)["gat"], obs[:, perm])
    yhat_p, _ = adapter(h[:, perm, :], y0[:, perm, :], s[:, perm], m_p)

    assert torch.allclose(yhat[:, perm, :], yhat_p, atol=1e-5)
