"""Kaggle entry point: unpack the code dataset, install it, check the environment, then run the skeleton."""
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

# Keep Kaggle's preinstalled CUDA torch: install our package and extras without touching it.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", f"{CODE}[stgnn,tsfm,patchtst]"], check=True)
subprocess.run(["nvidia-smi"])
env = dict(os.environ, OUT_DIR=OUT, PYTHONPATH=CODE)
with open(os.path.join(OUT, "env_check.txt"), "w") as f:
    check = subprocess.run([sys.executable, "-u", "scripts/env_check.py"], cwd=CODE, env=env,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    f.write(check.stdout)
print(check.stdout, flush=True)
skel = subprocess.run([sys.executable, "-u", "-m", "scripts.skeleton"], cwd=OUT, env=env)
sys.exit(check.returncode or skel.returncode)
