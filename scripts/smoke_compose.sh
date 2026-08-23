#!/usr/bin/env bash
# Live smoke test against a compose-booted Dewie container.
#
# Boots nothing itself — point it at a stack started from one of the
# docker-compose.e2e-*.yml files (all run with AUTH_ENABLED=false and the
# passthrough enrichment backend):
#
#   docker compose -f docker-compose.e2e-pgvector.yml up -d --build --wait
#   scripts/smoke_compose.sh 8767
#
# Drives the loop mocked tests can't: ingest a raw body over HTTP, wait for
# the enrichment worker to make it searchable, then read it back through both
# the MCP dispatch and the REST /api/query route.
#
# Usage: scripts/smoke_compose.sh [port]   (default 8767 = e2e-pgvector)
set -euo pipefail

PORT="${1:-8767}"

for _ in $(seq 1 30); do
    if curl -sf -m 2 "http://localhost:$PORT/health" >/dev/null 2>&1; then break; fi
    sleep 1
done
curl -sf -m 2 "http://localhost:$PORT/health" >/dev/null \
    || { echo "FAIL: no healthy server on :$PORT"; exit 1; }

SMOKE_PORT="$PORT" python3 - <<'EOF'
import json
import os
import sys
import time
import urllib.request

PORT = os.environ["SMOKE_PORT"]
BASE = f"http://localhost:{PORT}"


def post(path, payload):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
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
    {
        "url": "https://example.com/iceland-smoke",
        "title": "How Iceland monitors volcanic eruptions",
        "body": body,
    },
)
if status != 202 or not data.get("accepted"):
    fail("ingest", f"{status} {data}")
print("ok 1: ingest 202")

for _ in range(30):
    _, d = post(
        "/api/mcp",
        {"tool": "search_corpus", "input": {"query": "volcanic eruption monitoring iceland"}},
    )
    if d["content"]["count"] >= 1:
        break
    time.sleep(1)
else:
    fail("search", "ingested doc never became searchable")
print("ok 2: search_corpus finds the doc")

status, d = post("/api/query", {"query": "volcanic eruption monitoring iceland", "limit": 3})
if status != 200:
    fail("rest-query", f"{status} {d}")
titles = [r.get("title", "") for r in d.get("results", [])]
if not any("Iceland" in t for t in titles):
    fail("rest-query", f"doc missing from results: {titles}")
print("ok 3: REST /api/query returns the doc")

print("SMOKE PASS")
EOF
