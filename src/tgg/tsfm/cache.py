"""The encoder cache: shard writer, window_scale (Decision 6), and the manifest.

One cache directory holds one memory-mappable .npy per array of the interface (guide §6.2):
t (n,) int32, h (n, N, d) float16, y0 (n, N, 12) float32, s, loc, sd, obs (n, N), and h_last
(n, N, d) float16 for the bolt-L288 cache only (guide C7, the last patch token beside the mean).
"""
import json
import os

import numpy as np

from tgg.ckpt import sha256

ARRAYS = {"t": np.int32, "h": np.float16, "y0": np.float32, "s": np.float32,
          "loc": np.float32, "sd": np.float32, "obs": bool}
H_LAST = {"h_last": np.float16}


def window_scale(x: np.ndarray, m: np.ndarray, floor_mph: float = 1.0) -> np.ndarray:
    """(Decision 6) max(floor_mph, mean |x| over the observed lookback steps); floor_mph if none observed.

    x, m are (..., L) raw lookback values and mask; returns (...).
    """
    obs = m.astype(bool)
    cnt = obs.sum(-1)
    mean_abs = np.where(cnt > 0, np.abs(np.where(obs, x, 0)).sum(-1) / np.maximum(cnt, 1), 0.0)
    return np.maximum(mean_abs, floor_mph).astype(np.float32)


def create(cache_dir: str, n: int, N: int, d: int = 512, h_last: bool = False) -> dict[str, np.memmap]:
    """Create one open_memmap .npy per array of the cache row, ready to fill shard by shard.
    h_last=True also creates the (n, N, d) last-patch-token array (guide C7)."""
    os.makedirs(cache_dir, exist_ok=True)
    shapes = {"t": (n,), "h": (n, N, d), "y0": (n, N, 12), "s": (n, N),
              "loc": (n, N), "sd": (n, N), "obs": (n, N), "h_last": (n, N, d)}
    arrays = dict(ARRAYS, **(H_LAST if h_last else {}))
    return {name: np.lib.format.open_memmap(os.path.join(cache_dir, f"{name}.npy"), mode="w+",
                                             dtype=dtype, shape=shapes[name])
            for name, dtype in arrays.items()}


def open_memmap_dir(cache_dir: str) -> dict[str, np.memmap]:
    """Open an existing cache directory's arrays read-only, memory-mapped."""
    names = list(ARRAYS) + (["h_last"] if os.path.exists(os.path.join(cache_dir, "h_last.npy")) else [])
    return {name: np.load(os.path.join(cache_dir, f"{name}.npy"), mmap_mode="r") for name in names}


def write_manifest(cache_dir: str, git_sha: str, encoder: str, names: list[str] | None = None) -> dict:
    """sha256 per array plus the git sha and encoder id, written as manifest.json next to the arrays.
    `names` defaults to the base cache row; pass the keys actually written (e.g. from `create`) otherwise."""
    names = names if names is not None else list(ARRAYS)
    manifest = {"git_sha": git_sha, "encoder": encoder,
                "sha256": {name: sha256(os.path.join(cache_dir, f"{name}.npy")) for name in names}}
    with open(os.path.join(cache_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest
