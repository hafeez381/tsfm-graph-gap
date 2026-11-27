"""Kaggle entry point for job J12 + part of J13: all three adapter variants at 100%, 25% and 5%
on bolt-L12 (guide C16). Rebuilds the cache itself (as job J6 did): it's small enough and quick
enough on a T4 that uploading it as a separate dataset first is not worth the network time.

Fetch the output with `kaggle kernels output <user>/tgg-adapter-matrix-bolt-l12
-p results/kaggle/adapter-matrix-bolt-l12`.
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

import torch  # noqa: E402
pyg = f"https://data.pyg.org/whl/torch-{torch.__version__.split('+')[0]}+{torch.version.cuda and 'cu' + torch.version.cuda.replace('.', '') or 'cpu'}.html"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-f", pyg, "-e", f"{CODE}[stgnn,tsfm]"], check=True)
subprocess.run(["nvidia-smi"])

env = dict(os.environ, OUT_DIR=OUT, PYTHONPATH=CODE)
data_dir = os.path.join(OUT, "data")
build_data = subprocess.run([sys.executable, "-u", "-m", "scripts.build_data", "--dataset", "metr_la",
                              "--out", data_dir], cwd=CODE, env=env)
if build_data.returncode:
    sys.exit(build_data.returncode)

cache_root = os.path.join(OUT, "cache")
build_cache = subprocess.run(
    [sys.executable, "-u", "-m", "scripts.build_cache", "--mode", "full", "--split", "all",
     "--device", "cuda", "--data", data_dir, "--out", cache_root, "--results", os.path.join(OUT, "results")],
    cwd=CODE, env=env)
if build_cache.returncode:
    sys.exit(build_cache.returncode)

results = subprocess.run(
    [sys.executable, "-u", "-m", "scripts.run_adapter_matrix", "--cache", "bolt-L12",
     "--variants", "gat,mlp,full", "--fractions", "1.0,0.25,0.05", "--mode", "full",
     "--data", data_dir, "--cache-root", cache_root, "--results", os.path.join(OUT, "results")],
    cwd=CODE, env=env)
sys.exit(results.returncode)
