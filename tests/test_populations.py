import numpy as np

from tgg.eval import gamma, populations


def test_labels_on_crafted_windows():
    T, N = 40, 3
    X = np.full((T, N), 60.0, np.float32)
    M = np.ones((T, N), np.uint8)
    X[21:33, 0] = 20.0                           # sensor 0: free flow in the lookback, jam in the target
    X[21:33, 2] = 61.0                           # sensor 2: within 10% throughout
    v = np.full(N, 65.0, np.float32)
    onset, steady = populations.label(X, M, np.array([20]), v)
    assert onset[0].tolist() == [True, False, False]
    assert steady[0].tolist() == [False, True, True]
    assert populations.free_flow(X, M, 10)[1] == 60.0


def test_overlap_goes_to_onset():
    X = np.full((40, 1), 40.0, np.float32)
    M = np.ones((40, 1), np.uint8)
    X[21:33, 0] = 38.0                           # below 0.6 * 65 = 39 but within 10% of 40
    onset, steady = populations.label(X, M, np.array([20]), np.array([65.0], np.float32))
    assert onset[0, 0] and not steady[0, 0]


def test_gamma_positive_when_neighbour_leads():
    rng = np.random.default_rng(0)
    T = 900
    up = np.cumsum(rng.normal(0, 1, T)) + 60
    down = np.r_[np.full(6, 60.0), up[:-6]] + rng.normal(0, 0.1, T)   # sensor 1 follows sensor 0 six steps later
    X = np.stack([up, down], 1).astype(np.float32)
    M = np.ones_like(X, np.uint8)
    nbr = np.ones((2, 2), bool)
    g, _, _ = gamma.fit(X, X, M, nbr, np.arange(300, 600), np.arange(620, 880))
    assert g[1] > 0.1
