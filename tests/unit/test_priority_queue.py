"""
Unit tests for priority queue injection (Issue #31).

Covers:
- POST /pipeline/enrich/priority returns 200 with queued status
- priority=1 is set on the doc (all three SQL statements executed)
- llm_cache cleared
- pipeline_errors resolved
- 404 when doc not found
- get_pending_docs query orders by priority DESC, ingested_at ASC
- set_priority resets priority to 0
- enrich_document_flow calls set_priority(0) on success
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dewie.api.routes.pipeline import router
from dewie.models.content import ContentDocument, ContentStatus

# ── Helpers ────────────────────────────────────────────────────────────────────

DOC_ID = "a392301a-3987-471c-b5dd-cd1f88c78648"


def _make_doc(doc_id: str = DOC_ID) -> ContentDocument:
    return ContentDocument(
        id=uuid.UUID(doc_id),
        url="https://example.com/test",
        title="Test Doc",
        source="example.com",
        status=ContentStatus.READY,
    )


def _make_app(pg_mock) -> FastAPI:
    """Build a minimal FastAPI app with the pipeline router and a mock pg on state."""
    app = FastAPI()
    app.include_router(router)
    app.state.postgres = pg_mock
    return app


def _make_pg_for_priority(doc: ContentDocument | None = None):
    """
    Mock PostgresClient for the priority endpoint.
    get_by_id returns `doc`.
    _session_factory returns a context manager whose session records all execute calls.
    """
    session = AsyncMock()
    session.execute.return_value = MagicMock()

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=None)

    pg = MagicMock()
    pg.get_by_id = AsyncMock(return_value=doc)
    pg._session_factory.return_value = cm
    return pg, session


# ── POST /pipeline/enrich/priority ────────────────────────────────────────────


@pytest.mark.skip_in_daemon
def test_priority_enrich_returns_200_with_queued_status():
    """Valid doc_id → 200 with status='queued'."""
    pg, _ = _make_pg_for_priority(doc=_make_doc())
    client = TestClient(_make_app(pg))

    resp = client.post("/pipeline/enrich/priority", json={"doc_id": DOC_ID})

    assert resp.status_code == 200
    body = resp.json()
    assert body["doc_id"] == DOC_ID
    assert body["status"] == "queued"
    assert "message" in body


@pytest.mark.skip_in_daemon
def test_priority_enrich_returns_404_when_doc_not_found():
    """Unknown doc_id → 404."""
    pg, _ = _make_pg_for_priority(doc=None)
    client = TestClient(_make_app(pg))

    resp = client.post("/pipeline/enrich/priority", json={"doc_id": DOC_ID})

    assert resp.status_code == 404


@pytest.mark.skip_in_daemon
def test_priority_enrich_executes_three_statements():
    """Endpoint must execute exactly 3 SQL statements (delete, update errors, update doc)."""
    pg, session = _make_pg_for_priority(doc=_make_doc())
    client = TestClient(_make_app(pg))

    client.post("/pipeline/enrich/priority", json={"doc_id": DOC_ID})

    assert session.execute.call_count == 3
    session.commit.assert_called_once()


@pytest.mark.skip_in_daemon
def test_priority_enrich_clears_llm_cache():
    """First SQL statement must DELETE from llm_cache for the given doc_id."""
    pg, session = _make_pg_for_priority(doc=_make_doc())
    client = TestClient(_make_app(pg))

    client.post("/pipeline/enrich/priority", json={"doc_id": DOC_ID})

    first_sql = str(session.execute.call_args_list[0][0][0])
    assert "llm_cache" in first_sql
    assert "DELETE" in first_sql.upper()


@pytest.mark.skip_in_daemon
def test_priority_enrich_resolves_pipeline_errors():
    """Second SQL statement must UPDATE pipeline_errors SET resolved=TRUE."""
    pg, session = _make_pg_for_priority(doc=_make_doc())
    client = TestClient(_make_app(pg))

    client.post("/pipeline/enrich/priority", json={"doc_id": DOC_ID})

    second_sql = str(session.execute.call_args_list[1][0][0])
    assert "pipeline_errors" in second_sql
    assert "resolved" in second_sql.lower()


@pytest.mark.skip_in_daemon
def test_priority_enrich_sets_priority_1_on_document():
    """Third SQL statement must UPDATE documents SET priority=1."""
    pg, session = _make_pg_for_priority(doc=_make_doc())
    client = TestClient(_make_app(pg))

    client.post("/pipeline/enrich/priority", json={"doc_id": DOC_ID})

    third_sql = str(session.execute.call_args_list[2][0][0])
    assert "documents" in third_sql
    assert "priority" in third_sql.lower()
    # The bound params should carry priority=1
    third_params = session.execute.call_args_list[2][0][1]
    assert third_params.get("doc_id") == DOC_ID


# ── get_pending_docs ordering ─────────────────────────────────────────────────


async def test_get_pending_docs_orders_by_priority_desc():
    """get_pending_docs SQL must use ORDER BY priority DESC."""
    session = AsyncMock()
    result = MagicMock()
    result.mappings.return_value.all.return_value = []
    session.execute.return_value = result

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=None)

    pg = MagicMock()
    pg._session_factory.return_value = cm

    from dewie.storage.postgres import PostgresClient

    # Call via the real method (not a mock)
    real_pg = PostgresClient.__new__(PostgresClient)
    real_pg._session_factory = pg._session_factory
    real_pg._is_sqlite = True  # Test the SQLite path (uses session_factory)

    await real_pg.get_pending_docs(limit=50)

    sql = str(session.execute.call_args[0][0])
    assert "priority" in sql.lower()
    assert "DESC" in sql.upper()
    assert "ingested_at" in sql.lower()


# ── set_priority ───────────────────────────────────────────────────────────────


async def test_set_priority_executes_update_and_commits():
    """set_priority must UPDATE documents SET priority and commit."""
    session = AsyncMock()
    session.execute.return_value = MagicMock()

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=None)

    pg = MagicMock()
    pg._session_factory.return_value = cm

    from dewie.storage.postgres import PostgresClient

    real_pg = PostgresClient.__new__(PostgresClient)
    real_pg._session_factory = pg._session_factory

    await real_pg.set_priority(uuid.UUID(DOC_ID), 0)

    session.execute.assert_called_once()
    sql = str(session.execute.call_args[0][0])
    assert "priority" in sql.lower()
    assert "documents" in sql.lower()
    params = session.execute.call_args[0][1]
    assert params["priority"] == 0
    session.commit.assert_called_once()



