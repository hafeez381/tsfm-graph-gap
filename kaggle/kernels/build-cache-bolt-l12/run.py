"""Kaggle entry point for job J6: real METR-LA data, the frozen Bolt encoder, the bolt-L12 cache.

Downloads the kernel's code dataset, builds data/metr_la* if missing, then runs build_cache.py
--mode full for every split. Fetch the output with `kaggle kernels output <user>/tgg-build-cache-bolt-l12
-p results/kaggle/build-cache-bolt-l12`, then promote cache/bolt-L12 with kaggle/push_cache.sh.
"""
import glob
import os
import shutil
import subprocess
import sys
import tarfile

OUT, CODE = "/kaggle/working", "/tmp/code"

found = glob.glob("/kaggle/input/**/code.tgz", recursive=True)
if len(found) != 1:
    sys.exit(f"expected exactly one code.tgz under /kaggle/input, found {found}")
os.makedirs(CODE, exist_ok=True)
with tarfile.open(found[0]) as tar:
    tar.extractall(CODE, filter="data")
shutil.copy(os.path.join(os.path.dirname(found[0]), "COMMIT"), OUT)
print("commit", open(os.path.join(OUT, "COMMIT")).read().strip(), flush=True)

import torch  # noqa: E402  (preinstalled; its build picks the PyG wheel index, unused here but kept for parity)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", f"{CODE}[tsfm]"], check=True)
subprocess.run(["nvidia-smi"])

env = dict(os.environ, OUT_DIR=OUT, PYTHONPATH=CODE)
data_dir = os.path.join(OUT, "data")
if not os.path.exists(os.path.join(data_dir, "metr_la.npz")):
    build_data = subprocess.run([sys.executable, "-u", "-m", "scripts.build_data", "--dataset", "metr_la",
                                  "--out", data_dir], cwd=CODE, env=env)
    if build_data.returncode:
        sys.exit(build_data.returncode)

results = subprocess.run(
    [sys.executable, "-u", "-m", "scripts.build_cache", "--mode", "full", "--split", "all",
     "--device", "cuda", "--data", data_dir, "--out", os.path.join(OUT, "cache"),
     "--results", os.path.join(OUT, "results")],
    cwd=CODE, env=env)
sys.exit(results.returncode)
