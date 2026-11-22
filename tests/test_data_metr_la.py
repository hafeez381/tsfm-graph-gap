"""Checks on the built METR-LA files (run scripts.build_data first; skipped otherwise)."""
import os

import numpy as np
import pytest

from tgg.data import windows

D = "data"
pytestmark = pytest.mark.skipif(not os.path.exists(f"{D}/metr_la.npz"), reason="run scripts.build_data first")


def test_shapes_and_mask():
    d = np.load(f"{D}/metr_la.npz")
    X, M, Xf = d["X"], d["M"], d["X_filled"]
    assert X.shape == (34_272, 207)
    assert (X[M == 0] == 0).all() and (X[M == 1] > 0).all()
    assert not (Xf[M == 0] == 0).all()                  # X and X_filled really differ (Decision 1)
    assert d["tod"].min() == 0 and d["tod"].max() < 1 and (np.diff(d["ts"]) == 300).all()


def test_graph():
    g = np.load(f"{D}/metr_la_graph.npz")
    A, nbr = g["A"], g["nbr"]
    assert abs(int((A > 0).sum()) - 1515) <= 0.05 * 1515   # Decision 2 enforcement point
    assert np.median(nbr.sum(1)) <= 40
    assert nbr.diagonal().all() and (nbr == nbr.T).all()
    assert A[0, 37] > 0 and A[37, 0] == 0               # direction: j=0 upstream of i=37


def test_windows_and_fraction():
    w = np.load(f"{D}/metr_la_windows.npz")
    tr = w["train_t"]
    assert (tr + 12 < int(w["train_end"])).all()
    b = windows.fraction_block(tr, 0.05)
    assert len(b) == 1184 and b[-1] == tr[-1]
    with pytest.raises(AssertionError):
        windows.assert_fraction_block(b[1:], tr, 0.05)
