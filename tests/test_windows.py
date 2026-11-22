import numpy as np
import pytest

from tgg.data import windows


def test_splits_disjoint_and_inside_bounds():
    w = windows.make(34_272)
    tr, va, te = w["train_t"], w["val_t"], w["test_t"]
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
    assert tr.min() >= 287 and va.min() >= 287 and te.min() >= 287
    assert (tr + 12 < w["train_end"]).all()                     # every training target before validation
    assert (va + 1 >= w["train_end"]).all() and (va + 12 < w["val_end"]).all()
    assert (te + 1 >= w["val_end"]).all() and te.max() + 12 <= 34_271


def test_fraction_block_is_most_recent():
    tr = windows.make(34_272)["train_t"]
    b = windows.fraction_block(tr, 0.05)
    assert len(b) == int(np.floor(0.05 * len(tr))) and b[-1] == tr[-1]
    assert np.array_equal(b, tr[-len(b):])
    with pytest.raises(ValueError):
        windows.fraction_block(tr, 0.0)
