"""Kaggle entry point for job J14: PatchTST at 5%, 25% and 100% (guide C14).

Downloads the kernel's code dataset, builds data/metr_la* if missing, then runs train_patchtst.py
--mode full for each fraction in turn. Fetch the output with
`kaggle kernels output <user>/tgg-train-patchtst-l12 -p results/kaggle/train-patchtst-l12`.
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
# build_data.py needs tsl (stgnn extra); the model itself needs the patchtst extra.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-f", pyg, "-e", f"{CODE}[stgnn,patchtst]"],
               check=True)
subprocess.run(["nvidia-smi"])

env = dict(os.environ, OUT_DIR=OUT, PYTHONPATH=CODE)
data_dir = os.path.join(OUT, "data")
if not os.path.exists(os.path.join(data_dir, "metr_la.npz")):
    build_data = subprocess.run([sys.executable, "-u", "-m", "scripts.build_data", "--dataset", "metr_la",
                                  "--out", data_dir], cwd=CODE, env=env)
    if build_data.returncode:
        sys.exit(build_data.returncode)

for fraction in ("1.0", "0.25", "0.05"):
    r = subprocess.run(
        [sys.executable, "-u", "-m", "scripts.train_patchtst", "--fraction", fraction, "--mode", "full",
         "--data", data_dir, "--results", os.path.join(OUT, "results")],
        cwd=CODE, env=env)
    if r.returncode:
        sys.exit(r.returncode)
