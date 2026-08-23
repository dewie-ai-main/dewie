#!/usr/bin/env bash
# start.sh — Start Dewie
# Usage: ./start.sh [--no-docker]
#
# This script:
#   1. Starts Dewie's Docker services (Postgres, Redis)
#   2. Starts the Dewie API (port 8000)
#
# Prerequisites:
#   - Docker Desktop running (for Postgres/Redis)
#   - Python venv set up: uv sync or pip install -e .

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "⚡ Dewie startup"

# Activate venv
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "ERROR: .venv not found. Run: python -m venv .venv && pip install -e '.[dev]'"
    exit 1
fi

# 1. Start Docker services
if [[ "$1" != "--no-docker" ]]; then
    echo "→ Starting Docker services (Postgres, Redis)..."
    docker compose up -d
    echo "  Waiting for services..."
    sleep 5
    echo "  ✓ Docker services up"
fi

# 2. Start Dewie API
echo "→ Starting Dewie API on :8000..."
echo ""
python -m uvicorn dewie.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload


