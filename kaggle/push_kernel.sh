#!/usr/bin/env bash
# Push kaggle/kernels/<job>/ as the private GPU kernel <user>/tgg-<job>, attached to <user>/tgg-code
# plus any extra cache datasets named in $2.., space-separated slugs (without the account prefix).
# Usage: kaggle/push_kernel.sh <job> [extra-dataset-slug ...]    (run kaggle/build_code.sh first)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
source kaggle/common.sh
job="${1:?usage: kaggle/push_kernel.sh <job> [extra-dataset-slug ...]}"
shift
src="kaggle/kernels/$job"
[[ -f "$src/run.py" ]] || { echo "no $src/run.py" >&2; exit 1; }

sources="\"$KAGGLE_USER/tgg-code\""
for slug in "$@"; do
  sources="$sources, \"$KAGGLE_USER/$slug\""
done

tmp=$(mktemp -d)
cp "$src/run.py" "$tmp/"
cat > "$tmp/kernel-metadata.json" <<JSON
{"id": "$KAGGLE_USER/tgg-$job", "title": "tgg-$job", "code_file": "run.py", "language": "python",
 "kernel_type": "script", "is_private": "true", "enable_gpu": "true", "enable_tpu": "false",
 "enable_internet": "true", "dataset_sources": [$sources],
 "competition_sources": [], "kernel_sources": [], "model_sources": []}
JSON
(cd "$tmp" && "$KAGGLE" kernels push -p . --accelerator NvidiaTeslaT4)
echo "status: $KAGGLE kernels status $KAGGLE_USER/tgg-$job"
echo "output: $KAGGLE kernels output $KAGGLE_USER/tgg-$job -p results/kaggle/$job"
