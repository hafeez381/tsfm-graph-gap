"""Load a traffic dataset from tsl as raw speeds X (zeros = missing), mask M and forward-filled X_filled (Decision 1)."""
import numpy as np

DATASETS = {"metr_la": "MetrLA"}


def _tsl_dataset(name: str, root: str):
    import tsl.datasets

    return getattr(tsl.datasets, DATASETS[name])(root=root, impute_zeros=True)


def load(name: str, root: str) -> tuple[dict, object]:
    """Return the arrays of the data interface and the tsl dataset (for the graph).

    X (T, N) float32 raw speeds in mph, 0 where missing; M (T, N) uint8, 1 where observed;
    X_filled (T, N) float32 forward-filled (leading zeros stay 0); tod (T,) float32 minutes since
    midnight / 1440; ts (T,) int64 unix seconds.
    """
    ds = _tsl_dataset(name, root)
    X_filled = ds.dataframe().to_numpy(dtype=np.float32)
    M = ds.mask[..., 0].astype(np.uint8)
    X = X_filled * M                                   # tsl's mask is raw != 0, so this restores the raw zeros
    idx = ds.index
    tod = ((idx.hour * 60 + idx.minute) / 1440).to_numpy(dtype=np.float32)
    ts = (idx.tz_localize(None).asi8 // 10**9).astype(np.int64)
    assert X.shape == M.shape == X_filled.shape and not np.isnan(X_filled).any()
    assert (X[M == 0] == 0).all() and (X[M == 1] > 0).all()
    return {"X": X, "M": M, "X_filled": X_filled, "tod": tod, "ts": ts}, ds
