#!/usr/bin/env bash
# Upload a built cache directory as a new version of the private Kaggle dataset <user>/tgg-cache-<name>,
# with Israr and Abdul added as collaborators (guide C6: "Upload the cache as a private Kaggle dataset").
# Usage: kaggle/push_cache.sh <name> <cache_dir>
#   e.g. kaggle/push_cache.sh bolt-L12-metr cache/bolt-L12
# Needs ISRAR_KAGGLE_USER and ABDUL_KAGGLE_USER (their Kaggle usernames) to add them as collaborators;
# without them the dataset is still created/versioned, just not shared.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
source kaggle/common.sh
name="${1:?usage: kaggle/push_cache.sh <name> <cache_dir>}"
dir="${2:?usage: kaggle/push_cache.sh <name> <cache_dir>}"
[[ -d "$dir" ]] || { echo "no such cache dir: $dir" >&2; exit 1; }

slug="tgg-cache-$name"
collab="[]"
if [[ -n "${ISRAR_KAGGLE_USER:-}" && -n "${ABDUL_KAGGLE_USER:-}" ]]; then
  collab="[{\"username\": \"$ISRAR_KAGGLE_USER\", \"role\": \"reader\"}, {\"username\": \"$ABDUL_KAGGLE_USER\", \"role\": \"reader\"}]"
else
  echo "ISRAR_KAGGLE_USER / ABDUL_KAGGLE_USER not set: uploading without collaborators" >&2
fi

cat > "$dir/dataset-metadata.json" <<JSON
{"title": "$slug", "id": "$KAGGLE_USER/$slug", "licenses": [{"name": "CC0-1.0"}], "collaborators": $collab}
JSON

if "$KAGGLE" datasets status "$KAGGLE_USER/$slug" >/dev/null 2>&1; then
  "$KAGGLE" datasets version -p "$dir" -m "cache $name, $(git rev-parse HEAD)" -q -r zip
else
  "$KAGGLE" datasets create -p "$dir" -q -r zip
fi
echo "uploaded $KAGGLE_USER/$slug from $dir"
