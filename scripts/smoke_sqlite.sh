#!/usr/bin/env bash
# Live SQLite smoke test — the gate that catches what mocked unit tests can't.
#
# Boots a real server on SQLite, then drives the full loop:
#   1. POST /api/ingest with a raw body            → 202
#   2. doc becomes status=ready and searchable      → search_corpus finds it
#   3. web_search corpus-first                      → source=corpus, no web call
#   4. web_search on an absent topic (stub provider)→ source=web + auto-ingest
#   5. repeat the absent-topic query                → source=corpus (self-building)
#
# Usage: scripts/smoke_sqlite.sh [python-executable]
set -euo pipefail

PY="${1:-python3}"
PORT="${SMOKE_PORT:-18947}"
WORKDIR="$(mktemp -d)"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cleanup() {
    kill "$SERVER_PID" 2>/dev/null || true
    rm -rf "$WORKDIR"
}
trap cleanup EXIT

cd "$REPO"

STUB_RESULTS='[{"title":"Mount Etna activity report","url":"https://stub.example/etna","snippet":"etna","content":"Mount Etna in Sicily erupted in June 2026, sending ash plumes over Catania airport and disrupting flights. Lava flows from the southeast crater were monitored by INGV observatory teams using thermal cameras and drone surveys across the volcanic edifice throughout the eruption sequence."}]'

env -i HOME="$HOME" PATH="$PATH" \
    POSTGRES_DSN="sqlite+aiosqlite:///$WORKDIR/dewie.db" \
    AUTH_ENABLED=false \
    INTERNAL_SERVICE_KEY=smoke-key \
    SEARCH_PROVIDER=stub \
    DEWIE_STUB_SEARCH_RESULTS="$STUB_RESULTS" \
    DEWIE_DATA_DIR="$WORKDIR/bodies" \
    ENRICHMENT_SLEEP_SECS=1 \
    PYTHONPATH="$REPO/src" \
    "$PY" -m uvicorn dewie.main:app --port "$PORT" >"$WORKDIR/server.log" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 30); do
    if curl -s -m 2 "http://localhost:$PORT/api/mcp" >/dev/null 2>&1; then break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "FAIL: server died during startup"; tail -30 "$WORKDIR/server.log"; exit 1
    fi
    sleep 1
done

SMOKE_PORT="$PORT" "$PY" - <<'EOF'
import json
import os
import sys
import time
import urllib.request

PORT = os.environ["SMOKE_PORT"]
BASE = f"http://localhost:{PORT}"


def post(path, payload, headers=None):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.status, json.loads(resp.read())


def fail(step, detail):
    print(f"FAIL [{step}]: {detail}")
    sys.exit(1)


body = (
    "Volcanic monitoring in Iceland relies on a dense network of seismometers, "
    "continuous GPS stations, and gas sensors operated by the Icelandic "
    "Meteorological Office. When magma migrates beneath the Reykjanes peninsula, "
    "earthquake swarms and ground deformation are detected hours to days before "
    "an eruption begins, giving authorities time to evacuate towns like Grindavik. "
) * 4

status, data = post(
    "/api/ingest",
    {"url": "https://example.com/iceland", "title": "How Iceland monitors volcanic eruptions", "body": body},
    headers={"X-Service-Key": "smoke-key"},
)
if status != 202 or not data.get("accepted"):
    fail("ingest", f"{status} {data}")
print("ok 1: ingest 202")

# Wait for background enrichment to flip the doc to ready/searchable.
for _ in range(20):
    _, d = post("/api/mcp", {"tool": "search_corpus", "input": {"query": "volcanic eruption monitoring iceland"}})
    if d["content"]["count"] >= 1:
        break
    time.sleep(1)
else:
    fail("search", "ingested doc never became searchable")
print("ok 2: search_corpus finds the doc")

_, d = post("/api/mcp", {"tool": "web_search", "input": {"query": "volcanic eruption monitoring iceland"}})
if d["content"]["source"] != "corpus":
    fail("corpus-first", d["content"])
print("ok 3: web_search serves from corpus")

_, d = post("/api/mcp", {"tool": "web_search", "input": {"query": "mount etna eruption catania airport"}})
if d["content"]["source"] != "web" or not d["content"].get("ingested_doc_id"):
    fail("gap-fallback", d["content"])
print("ok 4: gap -> web fallback + auto-ingest")

for _ in range(20):
    _, d = post("/api/mcp", {"tool": "web_search", "input": {"query": "mount etna eruption catania airport"}})
    if d["content"]["source"] == "corpus":
        break
    time.sleep(1)
else:
    fail("self-building", f"repeat lookup never hit corpus: {d['content']}")
print("ok 5: repeat lookup served from corpus (self-building loop closed)")

print("SMOKE PASSED")
EOF
