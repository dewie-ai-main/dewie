"""Unit tests for /investigate and /investigate/jobs endpoints."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _disable_rate_limiting(monkeypatch):
    """Disable the module-level limiter during unit tests so rate-limit state
    from other tests doesn't bleed in and cause spurious 429 responses."""
    from dewie.api.middleware import limiter

    monkeypatch.setattr(limiter, "enabled", False)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_engine_mock(fetchone_result=None):
    """Return (pg, conn) with pg._engine mocked for begin/connect context managers."""
    mock_result = MagicMock()
    mock_result.fetchone.return_value = fetchone_result
    mock_result.fetchall.return_value = []

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=mock_result)

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_conn)
    cm.__aexit__ = AsyncMock(return_value=None)

    pg = AsyncMock()
    pg._engine = MagicMock()
    pg._engine.begin.return_value = cm
    pg._engine.connect.return_value = cm

    return pg, mock_conn, mock_result


def _make_investigate_app():
    """Minimal app with the investigate router."""
    from dewie.api.middleware import limiter
    from dewie.api.routes.investigate import router

    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(router)
    return app


def _make_investigate_v2_app(pg=None):
    """Minimal app with the investigate_v2 router."""
    from dewie.api.middleware import limiter
    from dewie.api.routes.investigate_v2 import router

    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(router)

    if pg is None:
        pg, _, _ = _make_engine_mock()
    app.state.postgres = pg
    return app


# ── POST /investigate ─────────────────────────────────────────────────────────


class TestInvestigateEndpoint:
    def test_missing_query_returns_422(self):
        client = TestClient(_make_investigate_app(), raise_server_exceptions=False)
        resp = client.post("/investigate", json={})
        assert resp.status_code == 422

    def test_empty_query_returns_422(self):
        client = TestClient(_make_investigate_app(), raise_server_exceptions=False)
        resp = client.post("/investigate", json={"query": ""})
        assert resp.status_code == 422

    def test_query_too_long_returns_422(self):
        client = TestClient(_make_investigate_app(), raise_server_exceptions=False)
        resp = client.post("/investigate", json={"query": "x" * 2001})
        assert resp.status_code == 422

    def test_happy_path_returns_200(self):
        """Full pipeline mocked — validates response shape."""
        client = TestClient(_make_investigate_app(), raise_server_exceptions=False)

        with (
            patch(
                "dewie.api.routes.investigate._decompose", AsyncMock(return_value=["sub q 1"])
            ),
            patch(
                "dewie.api.routes.investigate._search_all",
                AsyncMock(return_value={"sub q 1": []}),
            ),
            patch("dewie.api.routes.investigate._aggregate", return_value={}),
            patch(
                "dewie.api.routes.investigate._synthesize",
                AsyncMock(return_value="Final report text."),
            ),
        ):
            resp = client.post("/investigate", json={"query": "What are the best cloud databases?"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "What are the best cloud databases?"
        assert data["report"] == "Final report text."
        assert "sub_questions" in data
        assert "sources" in data
        assert "trace" in data
        assert "total_facts" in data

    def test_aq_not_exposed_in_response(self):
        """answers_questions must never appear in investigate response."""
        client = TestClient(_make_investigate_app(), raise_server_exceptions=False)

        with (
            patch("dewie.api.routes.investigate._decompose", AsyncMock(return_value=["q"])),
            patch(
                "dewie.api.routes.investigate._search_all", AsyncMock(return_value={"q": []})
            ),
            patch("dewie.api.routes.investigate._aggregate", return_value={}),
            patch("dewie.api.routes.investigate._synthesize", AsyncMock(return_value="answer")),
        ):
            resp = client.post("/investigate", json={"query": "test"})

        assert resp.status_code == 200
        assert "answers_questions" not in resp.text


# ── POST /investigate/jobs ────────────────────────────────────────────────────


class TestInvestigateJobsCreate:
    def test_create_job_returns_pending(self):
        job_id = str(uuid.uuid4())
        now = datetime.now(UTC)
        pg, _, mock_result = _make_engine_mock(fetchone_result=(uuid.UUID(job_id), now))

        client = TestClient(_make_investigate_v2_app(pg), raise_server_exceptions=False)

        with patch("asyncio.create_task"):
            resp = client.post(
                "/investigate/jobs",
                json={"query": "best vacation markets", "strategy": "matrix"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "pending"
        assert data["query"] == "best vacation markets"
        assert "id" in data
        assert "created_at" in data

    def test_missing_query_returns_422(self):
        client = TestClient(_make_investigate_v2_app(), raise_server_exceptions=False)
        resp = client.post("/investigate/jobs", json={"strategy": "matrix"})
        assert resp.status_code == 422

    def test_invalid_strategy_returns_422(self):
        client = TestClient(_make_investigate_v2_app(), raise_server_exceptions=False)
        resp = client.post(
            "/investigate/jobs",
            json={"query": "test", "strategy": "invalid_strategy"},
        )
        assert resp.status_code == 422

    def test_valid_strategies_accepted(self):
        for strategy in ("matrix", "subquestion", "plan"):
            job_id = str(uuid.uuid4())
            now = datetime.now(UTC)
            pg, _, _ = _make_engine_mock(fetchone_result=(uuid.UUID(job_id), now))
            client = TestClient(_make_investigate_v2_app(pg), raise_server_exceptions=False)
            with patch("asyncio.create_task"):
                resp = client.post(
                    "/investigate/jobs",
                    json={"query": "test", "strategy": strategy},
                )
            assert resp.status_code == 200, f"strategy={strategy} returned {resp.status_code}"


# ── GET /investigate/jobs/{job_id} ────────────────────────────────────────────


class TestInvestigateJobsGet:
    def _app_with_job(self, job: dict) -> tuple[FastAPI, TestClient]:
        pg, mock_conn, mock_result = _make_engine_mock()

        # _fetch_job returns a row based on fetchone result
        mock_row = MagicMock()
        for k, v in job.items():
            setattr(mock_row, k, v)
        mock_result.fetchone.return_value = mock_row

        app = _make_investigate_v2_app(pg)
        return app, TestClient(app, raise_server_exceptions=False)

    def test_pending_job_returns_status(self):
        job_id = str(uuid.uuid4())
        now_str = datetime.now(UTC).isoformat()

        with patch(
            "dewie.api.routes.investigate_v2._fetch_job",
            AsyncMock(
                return_value={
                    "id": job_id,
                    "query": "test query",
                    "strategy": "matrix",
                    "status": "pending",
                    "plan": None,
                    "result": None,
                    "error": None,
                    "created_at": now_str,
                    "started_at": None,
                    "completed_at": None,
                }
            ),
        ):
            pg, _, _ = _make_engine_mock()
            client = TestClient(_make_investigate_v2_app(pg), raise_server_exceptions=False)
            resp = client.get(f"/investigate/jobs/{job_id}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == job_id
        assert data["status"] == "pending"
        assert data["query"] == "test query"

    def test_missing_job_returns_404(self):
        with patch(
            "dewie.api.routes.investigate_v2._fetch_job",
            AsyncMock(return_value=None),
        ):
            pg, _, _ = _make_engine_mock()
            client = TestClient(_make_investigate_v2_app(pg), raise_server_exceptions=False)
            resp = client.get(f"/investigate/jobs/{uuid.uuid4()}")

        assert resp.status_code == 404

    def test_done_job_includes_result(self):
        job_id = str(uuid.uuid4())
        now_str = datetime.now(UTC).isoformat()

        with patch(
            "dewie.api.routes.investigate_v2._fetch_job",
            AsyncMock(
                return_value={
                    "id": job_id,
                    "query": "done query",
                    "strategy": "subquestion",
                    "status": "done",
                    "plan": None,
                    "result": {"report": "Final answer here."},
                    "error": None,
                    "created_at": now_str,
                    "started_at": now_str,
                    "completed_at": now_str,
                }
            ),
        ):
            pg, _, _ = _make_engine_mock()
            client = TestClient(_make_investigate_v2_app(pg), raise_server_exceptions=False)
            resp = client.get(f"/investigate/jobs/{job_id}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "done"
        assert data["result"]["report"] == "Final answer here."


# ── Logging Tests ─────────────────────────────────────────────────────────────

import logging


class TestInvestigateLogging:
    def test_investigate_logs_start_with_request_id(self, caplog):
        """Investigate endpoint logs 'investigate started' with request_id."""
        with caplog.at_level(logging.INFO, logger="dewie.api"):
            client = TestClient(_make_investigate_app(), raise_server_exceptions=False)
            with (
                patch(
                    "dewie.api.routes.investigate._decompose",
                    AsyncMock(return_value=["sub q 1"]),
                ),
                patch(
                    "dewie.api.routes.investigate._search_all",
                    AsyncMock(return_value={"sub q 1": []}),
                ),
                patch("dewie.api.routes.investigate._aggregate", return_value={}),
                patch(
                    "dewie.api.routes.investigate._synthesize",
                    AsyncMock(return_value="Final report."),
                ),
            ):
                resp = client.post(
                    "/investigate",
                    json={"query": "best cloud databases", "model": "test-model"},
                )

        assert resp.status_code == 200
        start_records = [r for r in caplog.records if r.message.startswith("investigate started")]
        assert len(start_records) >= 1
        log_entry = start_records[0]
        assert log_entry.request_id == "unknown"
        assert "best cloud databases" in log_entry.query

    def test_investigate_logs_success_with_timing(self, caplog):
        """Investigate endpoint logs 'investigate succeeded' with timing info."""
        with caplog.at_level(logging.INFO, logger="dewie.api"):
            client = TestClient(_make_investigate_app(), raise_server_exceptions=False)
            with (
                patch(
                    "dewie.api.routes.investigate._decompose",
                    AsyncMock(return_value=["sub q 1"]),
                ),
                patch(
                    "dewie.api.routes.investigate._search_all",
                    AsyncMock(return_value={"sub q 1": []}),
                ),
                patch("dewie.api.routes.investigate._aggregate", return_value={}),
                patch(
                    "dewie.api.routes.investigate._synthesize",
                    AsyncMock(return_value="Final report."),
                ),
            ):
                resp = client.post(
                    "/investigate",
                    json={"query": "test query", "num_sources": 5},
                )

        assert resp.status_code == 200
        success_records = [r for r in caplog.records if r.message.startswith("investigate succeeded")]
        assert len(success_records) >= 1
        log_entry = success_records[0]
        assert log_entry.request_id == "unknown"
        assert log_entry.status == 200
        assert isinstance(log_entry.elapsed_seconds, (int, float))

    def test_investigate_logs_error_on_exception(self, caplog):
        """Investigate endpoint logs exception details on failure."""
        with caplog.at_level(logging.ERROR, logger="dewie.api"):
            client = TestClient(_make_investigate_app(), raise_server_exceptions=False)
            with patch(
                "dewie.api.routes.investigate._decompose",
                AsyncMock(side_effect=RuntimeError("decompose failed")),
            ):
                resp = client.post(
                    "/investigate",
                    json={"query": "failing query"},
                )

        assert resp.status_code == 500
        fail_records = [r for r in caplog.records if r.message.startswith("investigate failed")]
        assert len(fail_records) >= 1
        log_entry = fail_records[0]
        assert log_entry.request_id == "unknown"
        assert "failing query" in log_entry.query

    def test_investigate_redacts_api_key_in_log(self, caplog):
        """Investigate endpoint redacts sensitive fields in request body."""
        from dewie.api.routes.investigate import _redact_fields

        body = {"query": "test", "api_key": "sk-secret-123", "model": "gpt-4"}
        redacted = _redact_fields(body)
        assert redacted["api_key"] == "***REDACTED***"
        assert redacted["query"] == "test"
        assert redacted["model"] == "gpt-4"

    def test_investigate_truncates_long_bodies(self, caplog):
        """Investigate endpoint truncates request bodies to 1000 chars."""
        from dewie.api.routes.investigate import _truncate

        long_str = "x" * 2000
        result = _truncate(long_str)
        assert len(result) < 2000
        assert "..." in result
        assert "2000 chars total" in result

        short = _truncate("short")
        assert short == "short"


class TestInvestigateV2Logging:
    def test_create_job_logs_start(self, caplog):
        """create_investigate_job logs 'create_investigate_job started'."""
        with caplog.at_level(logging.INFO, logger="dewie.api"):
            job_id = str(uuid.uuid4())
            now = datetime.now(UTC)
            pg, _, mock_result = _make_engine_mock(fetchone_result=(uuid.UUID(job_id), now))
            client = TestClient(_make_investigate_v2_app(pg), raise_server_exceptions=False)
            with patch("asyncio.create_task"):
                resp = client.post(
                    "/investigate/jobs",
                    json={"query": "best vacation markets", "strategy": "matrix"},
                )

        assert resp.status_code == 200
        start_records = [r for r in caplog.records if r.message.startswith("create_investigate_job started")]
        assert len(start_records) >= 1
        log_entry = start_records[0]
        assert log_entry.request_id == "unknown"
        assert "best vacation markets" in log_entry.query
        assert log_entry.strategy == "matrix"

    def test_create_job_logs_success_with_job_id(self, caplog):
        """create_investigate_job logs 'create_investigate_job succeeded' with job_id."""
        with caplog.at_level(logging.INFO, logger="dewie.api"):
            job_id = str(uuid.uuid4())
            now = datetime.now(UTC)
            pg, _, _ = _make_engine_mock(fetchone_result=(uuid.UUID(job_id), now))
            client = TestClient(_make_investigate_v2_app(pg), raise_server_exceptions=False)
            with patch("asyncio.create_task"):
                resp = client.post(
                    "/investigate/jobs",
                    json={"query": "test query", "strategy": "subquestion"},
                )

        assert resp.status_code == 200
        success_records = [r for r in caplog.records if r.message.startswith("create_investigate_job succeeded")]
        assert len(success_records) >= 1
        log_entry = success_records[0]
        assert log_entry.request_id == "unknown"
        assert log_entry.job_id == job_id
        assert log_entry.status == 200
        assert isinstance(log_entry.elapsed_seconds, (int, float))

    def test_create_job_redacts_sensitive_fields(self, caplog):
        """create_investigate_job endpoint redacts sensitive fields."""
        from dewie.api.routes.investigate_v2 import _redact_fields

        body = {"query": "test", "api_key": "secret-key", "token": "bearer-123"}
        redacted = _redact_fields(body)
        assert redacted["api_key"] == "***REDACTED***"
        assert redacted["token"] == "***REDACTED***"
        assert redacted["query"] == "test"

    def test_v2_truncation_helper(self, caplog):
        """investigate_v2 _truncate works correctly."""
        from dewie.api.routes.investigate_v2 import _truncate

        result = _truncate("x" * 1500)
        assert "..." in result
        assert "1500 chars total" in result
        assert _truncate("hello") == "hello"
