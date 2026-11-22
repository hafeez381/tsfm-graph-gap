# Sourced by the Kaggle scripts: the CLI path and the account that owns the code dataset and kernels.
KAGGLE="${KAGGLE:-$(command -v kaggle || echo "$HOME/.local/bin/kaggle")}"
KAGGLE_USER="${KAGGLE_USER:-$("$KAGGLE" config view 2>/dev/null | sed -n 's/^- username: //p')}"
[[ -n "$KAGGLE_USER" ]] || { echo "set KAGGLE_USER or log in with 'kaggle auth login'" >&2; exit 1; }
