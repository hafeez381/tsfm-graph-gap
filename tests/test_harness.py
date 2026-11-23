import json

import numpy as np
import pytest

from tgg.baselines import naive
from tgg.data import windows
from tgg.eval.harness import score, targets

T, N = 900, 6


@pytest.fixture
def data_dir(tmp_path):
    rng = np.random.default_rng(0)
    X = (60 + rng.normal(0, 3, (T, N))).astype(np.float32)
    M = (rng.random((T, N)) > 0.1).astype(np.uint8)
    M[650:670] = 0                                        # a network-wide outage longer than the lookback
    X *= M
    ts = (1_330_560_000 + 300 * np.arange(T)).astype(np.int64)
    np.savez(tmp_path / "toy.npz", X=X, M=M, X_filled=X, tod=np.zeros(T, np.float32), ts=ts)
    np.savez(tmp_path / "toy_windows.npz", **windows.make(T))
    return tmp_path


def _write(path, t, yhat):
    np.savez(path, t=t, yhat=yhat.astype(np.float32))
    return str(path)


def test_perfect_predictions_score_zero(data_dir):
    d, w = np.load(data_dir / "toy.npz"), np.load(data_dir / "toy_windows.npz")
    y, _ = targets(d["X"], d["M"], w["test_t"])
    m = score(_write(data_dir / "p.npz", w["test_t"], y), "toy", "test", str(data_dir))
    assert m["populations"]["all"]["mae_avg"] == 0 and m["populations"]["all"]["rmse_avg"] == 0


def test_missing_window_raises(data_dir):
    w = np.load(data_dir / "toy_windows.npz")
    t = w["test_t"][1:]
    with pytest.raises(AssertionError):
        score(_write(data_dir / "p.npz", t, np.zeros((len(t), N, 12))), "toy", "test", str(data_dir))


def test_day_sums_add_up_and_exclusion_counted(data_dir):
    w = np.load(data_dir / "toy_windows.npz")
    t = w["val_t"]
    m = score(_write(data_dir / "v.npz", t, np.full((len(t), N, 12), 55.0)), "toy", "val", str(data_dir))
    s = np.load(data_dir / "val_day_sums.npz")
    assert s["cnt"][0].sum() == m["populations"]["all"]["n_targets"]
    assert np.isclose(s["abs"][0].sum() / s["cnt"][0].sum(), m["populations"]["all"]["mae_avg"])
    assert m["n_excluded_no_lookback"] > 0                # the outage empties some lookbacks
    assert json.load(open(data_dir / "val_metrics.json"))["split"] == "val"


def test_last_value_and_ha(data_dir):
    d, w = np.load(data_dir / "toy.npz"), np.load(data_dir / "toy_windows.npz")
    X, M, ts = d["X"], d["M"], d["ts"]
    table = naive.historical_average(X, M, ts, int(w["train_end"]))
    t = w["test_t"][:5]
    lv = naive.last_value(X, M, ts, t, table)
    i = 0
    obs = np.flatnonzero(M[t[0] - 11:t[0] + 1, i])
    assert lv[0, i, 0] == X[t[0] - 11 + obs[-1], i] and (lv[0, i] == lv[0, i, 0]).all()
    with pytest.raises(AssertionError):
        naive.assert_train_only_fit(np.arange(int(w["train_end"]) + 1), int(w["train_end"]))
    assert naive.week_slot(np.array([1_330_560_000]))[0] == 3 * 288   # 2012-03-01 00:00 UTC was a Thursday
