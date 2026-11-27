"""Kaggle entry point: rerun C12's GAT adapter at 100% and 5% on bolt-L12, on code that descends
from the v-prereg tag (guide C5), so meta.json's git sha satisfies C16's verification check.

Builds data and the bolt-L12 cache (same path as job J6), then trains. Fetch the output with
`kaggle kernels output <user>/tgg-rerun-c12-adapter -p results/kaggle/rerun-c12-adapter`.
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
if not os.path.exists(os.path.join(data_dir, "metr_la.npz")):
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

for fraction in ("1.0", "0.05"):
    r = subprocess.run(
        [sys.executable, "-u", "-m", "scripts.train_adapter", "--cache", "bolt-L12", "--variant", "gat",
         "--fraction", fraction, "--mode", "full", "--data", data_dir, "--cache-root", cache_root,
         "--results", os.path.join(OUT, "results")],
        cwd=CODE, env=env)
    if r.returncode:
        sys.exit(r.returncode)
