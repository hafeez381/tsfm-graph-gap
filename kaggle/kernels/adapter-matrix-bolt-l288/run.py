"""Kaggle entry point for the rest of job J13: all three adapter variants at 100%, 25% and 5%
on bolt-L288 (guide C16). Pulls the cache's three splits from their own datasets.

Fetch the output with `kaggle kernels output <user>/tgg-adapter-matrix-bolt-l288
-p results/kaggle/adapter-matrix-bolt-l288`.
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
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-f", pyg, "-e", f"{CODE}[stgnn]"], check=True)
subprocess.run(["nvidia-smi"])

env = dict(os.environ, OUT_DIR=OUT, PYTHONPATH=CODE)
data_dir = os.path.join(OUT, "data")
build_data = subprocess.run([sys.executable, "-u", "-m", "scripts.build_data", "--dataset", "metr_la",
                              "--out", data_dir], cwd=CODE, env=env)
if build_data.returncode:
    sys.exit(build_data.returncode)

cache_root = os.path.join(OUT, "cache")
os.makedirs(os.path.join(cache_root, "bolt-L288"), exist_ok=True)
for split in ("train", "val", "test"):
    hit = glob.glob(f"/kaggle/input/*cache-bolt-l288-{split}*")
    if len(hit) != 1:
        sys.exit(f"expected exactly one bolt-L288 {split} cache dataset, found {hit}")
    os.symlink(os.path.join(hit[0], split), os.path.join(cache_root, "bolt-L288", split))

results = subprocess.run(
    [sys.executable, "-u", "-m", "scripts.run_adapter_matrix", "--cache", "bolt-L288",
     "--variants", "gat,mlp,full", "--fractions", "1.0,0.25,0.05", "--mode", "full",
     "--data", data_dir, "--cache-root", cache_root, "--results", os.path.join(OUT, "results")],
    cwd=CODE, env=env)
sys.exit(results.returncode)
