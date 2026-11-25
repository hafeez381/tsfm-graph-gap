import json

import numpy as np

from tgg.tsfm import cache


def test_window_scale_floor_and_mean():
    x = np.array([[10.0, 20.0, 0.0], [0.0, 0.0, 0.0], [0.4, 0.4, 0.4]])
    m = np.array([[1, 1, 0], [0, 0, 0], [1, 1, 1]])
    s = cache.window_scale(x, m)
    assert np.isclose(s[0], 15.0)                 # mean(|10|, |20|) over the 2 observed steps
    assert s[1] == 1.0                             # no observed step -> the floor
    assert s[2] == 1.0                              # mean 0.4 is below the floor -> the floor wins


def test_create_write_manifest_round_trip(tmp_path):
    cache_dir = str(tmp_path / "bolt-L12" / "test")
    n, N, d = 5, 3, 8
    arrays = cache.create(cache_dir, n, N, d)
    assert set(arrays) == set(cache.ARRAYS)
    arrays["t"][:] = np.arange(n, dtype=np.int32)
    arrays["h"][:] = np.ones((n, N, d), dtype=np.float16)
    arrays["obs"][:] = True
    for a in arrays.values():
        a.flush()

    manifest = cache.write_manifest(cache_dir, "deadbeef", "bolt-L12")
    assert manifest["git_sha"] == "deadbeef" and manifest["encoder"] == "bolt-L12"
    assert set(manifest["sha256"]) == set(cache.ARRAYS)
    on_disk = json.load(open(f"{cache_dir}/manifest.json"))
    assert on_disk == manifest

    reopened = cache.open_memmap_dir(cache_dir)
    assert np.array_equal(reopened["t"], np.arange(n, dtype=np.int32))
    assert reopened["h"].shape == (n, N, d) and reopened["h"].dtype == np.float16
    assert reopened["obs"].all()
