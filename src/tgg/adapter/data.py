"""CacheDataset: one item = one window, all N sensors (guide C12)."""
import os

import numpy as np
from torch.utils.data import Dataset

from tgg.tsfm import cache

H = 12


def assert_ids_in_split(t_all: np.ndarray, ids: np.ndarray, pos: np.ndarray) -> None:
    """Raise unless every requested window id is exactly present in the split's cache, in order."""
    if not np.array_equal(t_all[pos], ids):
        raise AssertionError("requested window ids are not a subset of this split's cache")


class CacheDataset(Dataset):
    """A memory-mapped encoder cache (guide §6.2), restricted to `ids` (e.g. a fraction_block of the
    training split; None for the whole split). One item holds `h` (N, d), `y0` (N, H), `s` (N,),
    `obs` (N,), and the target `y` (N, H) and mask `m` (N, H) from the raw X, M (never X_filled)."""

    def __init__(self, cache_dir: str, split: str, X: np.ndarray, M: np.ndarray, ids: np.ndarray | None = None):
        self.arrays = cache.open_memmap_dir(os.path.join(cache_dir, split))
        t_all = np.asarray(self.arrays["t"])
        if ids is None:
            self.pos = np.arange(len(t_all))
        else:
            self.pos = np.searchsorted(t_all, ids)
            assert_ids_in_split(t_all, ids, self.pos)
        self.t = t_all
        self.X, self.M = X, M

    def __len__(self) -> int:
        return len(self.pos)

    def __getitem__(self, idx: int) -> dict:
        p = self.pos[idx]
        t = int(self.t[p])
        future = t + np.arange(1, H + 1)
        obs = np.asarray(self.arrays["obs"][p])
        y = self.X[future].T.astype(np.float32)                      # (N, H), raw X, never X_filled
        m = self.M[future].T.astype(bool) & obs[:, None]             # also excluded if the lookback was empty
        return {
            "h": np.asarray(self.arrays["h"][p]).astype(np.float32),
            "y0": np.asarray(self.arrays["y0"][p]).astype(np.float32),
            "s": np.asarray(self.arrays["s"][p]).astype(np.float32),
            "obs": obs,
            "y": y,
            "m": m,
            "t": t,
        }
