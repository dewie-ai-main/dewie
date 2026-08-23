# Contributing to Dewie

Thanks for your interest. This project runs on a simple principle: **trust, but verify** — every claim about behavior is backed by a test or a live check, and CI enforces it.

## Ground rules

1. **CI must be green.** Every PR runs three gates:
   - `ruff check src tests` — zero tolerance, the repo is lint-clean
   - the unit + e2e suites (mocked storage, ~30s)
   - the **live SQLite smoke** (`scripts/smoke_sqlite.sh`) — boots a real server and drives ingest → search → corpus-first `web_search` → self-build. Mocked tests cannot catch wiring bugs; this can.
2. **A red suite is a stop-the-line event.** Don't merge on top of failures; fix or revert.
3. **No Postgres-only SQL in shared paths.** SQLite is a first-class deployment target. If you write SQL that runs on both engines, test both (`_session_is_sqlite` helpers exist in `storage/rankers.py`).
4. **Telemetry never fails the request it observes.** Audit logs, query logs, and metrics writes must be non-fatal (`try/except` + warning). There is history here.
5. **`answers_questions` never leaves the server.** It is an internal ranking signal — never include it in API responses, MCP tool outputs, or search result payloads.

## Dev setup

```bash
pip install -e ".[dev]"
pytest tests/unit tests/e2e -q --no-cov   # fast suites
ruff check src tests
./scripts/smoke_sqlite.sh                  # the real-server gate
```

Live-service tests are marked and deselected by default (`integration`, `production`, `perf`). Run them explicitly: `pytest -m integration`.

## Writing tests

- A failing test means one of two things: the code is wrong (fix the code) or the test describes an interface that never existed (delete the test, say so in the PR).
- Don't assert on your local config (`dewie.yml` contents, env vars) — patch what you depend on. The conftest scrubs leaked env state for you.
- Don't write tests that hit live servers from the unit suite; mark them `integration`.

## PR checklist

- [ ] Three CI gates green
- [ ] New SQL works on Postgres *and* SQLite (or is explicitly engine-gated)
- [ ] New env vars documented in `docs/configuration.md`
- [ ] New MCP tools documented in `docs/mcp-tools.md` and added to the manifest test
- [ ] No secrets, internal hostnames, or personal paths in code or fixtures
