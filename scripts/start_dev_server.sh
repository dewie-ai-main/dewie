#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

cd "$REPO_DIR"

if [ -f "$REPO_DIR/.env" ]; then
    set -a
    source "$REPO_DIR/.env"
    set +a
fi

# --reload-dir src: watch only source. Without it uvicorn watches the whole
# repo — and if the server log is appended inside the repo, every watcher
# log line triggers another change event (a 910k-line feedback loop).
exec "$REPO_DIR/.venv/bin/python3" -m uvicorn dewie.main:app \
    --host 0.0.0.0 \
    --port "${PORT:-10946}" \
    --reload \
    --reload-dir "$REPO_DIR/src"
