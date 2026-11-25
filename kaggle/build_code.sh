#!/usr/bin/env bash
# Package the committed code as a new version of the private Kaggle dataset <user>/tgg-code.
# Usage: kaggle/build_code.sh   (from anywhere in the repo, with a clean working tree)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
source kaggle/common.sh

if [[ -n "$(git status --porcelain)" ]]; then
  echo "working tree not clean: commit first, so results trace to a commit" >&2
  exit 1
fi

sha=$(git rev-parse HEAD)
git archive HEAD | gzip > kaggle/code/code.tgz
echo "$sha" > kaggle/code/COMMIT
cat > kaggle/code/dataset-metadata.json <<JSON
{"title": "tgg-code", "id": "$KAGGLE_USER/tgg-code", "licenses": [{"name": "CC0-1.0"}]}
JSON

# cd into the folder and pass "." rather than "kaggle/code": the forward slash in that relative
# path breaks the Kaggle CLI's upload temp-file naming on Windows (kaggle 2.2.5).
if (cd kaggle/code && "$KAGGLE" datasets status "$KAGGLE_USER/tgg-code" >/dev/null 2>&1); then
  (cd kaggle/code && "$KAGGLE" datasets version -p . -m "$sha" -q)
else
  (cd kaggle/code && "$KAGGLE" datasets create -p . -q)
fi
# Kernels attach the latest version, so wait until this one is ready before pushing a kernel.
sleep 15
until [[ "$("$KAGGLE" datasets status "$KAGGLE_USER/tgg-code" 2>/dev/null)" == *ready* ]]; do sleep 10; done
echo "uploaded $sha"
