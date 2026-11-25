"""The encoder cache: shard writer, window_scale (Decision 6), and the manifest.

One cache directory holds one memory-mappable .npy per array of the interface (guide §6.2):
t (n,) int32, h (n, N, d) float16, y0 (n, N, 12) float32, s, loc, sd, obs (n, N).
"""
import json
import os

import numpy as np

from tgg.ckpt import sha256

ARRAYS = {"t": np.int32, "h": np.float16, "y0": np.float32, "s": np.float32,
          "loc": np.float32, "sd": np.float32, "obs": bool}


def window_scale(x: np.ndarray, m: np.ndarray, floor_mph: float = 1.0) -> np.ndarray:
    """(Decision 6) max(floor_mph, mean |x| over the observed lookback steps); floor_mph if none observed.

    x, m are (..., L) raw lookback values and mask; returns (...).
    """
    obs = m.astype(bool)
    cnt = obs.sum(-1)
    mean_abs = np.where(cnt > 0, np.abs(np.where(obs, x, 0)).sum(-1) / np.maximum(cnt, 1), 0.0)
    return np.maximum(mean_abs, floor_mph).astype(np.float32)


def create(cache_dir: str, n: int, N: int, d: int = 512) -> dict[str, np.memmap]:
    """Create one open_memmap .npy per array of the cache row, ready to fill shard by shard."""
    os.makedirs(cache_dir, exist_ok=True)
    shapes = {"t": (n,), "h": (n, N, d), "y0": (n, N, 12), "s": (n, N),
              "loc": (n, N), "sd": (n, N), "obs": (n, N)}
    return {name: np.lib.format.open_memmap(os.path.join(cache_dir, f"{name}.npy"), mode="w+",
                                             dtype=dtype, shape=shapes[name])
            for name, dtype in ARRAYS.items()}


def open_memmap_dir(cache_dir: str) -> dict[str, np.memmap]:
    """Open an existing cache directory's arrays read-only, memory-mapped."""
    return {name: np.load(os.path.join(cache_dir, f"{name}.npy"), mmap_mode="r") for name in ARRAYS}


def write_manifest(cache_dir: str, git_sha: str, encoder: str) -> dict:
    """sha256 per array plus the git sha and encoder id, written as manifest.json next to the arrays."""
    manifest = {"git_sha": git_sha, "encoder": encoder,
                "sha256": {name: sha256(os.path.join(cache_dir, f"{name}.npy")) for name in ARRAYS}}
    with open(os.path.join(cache_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest
