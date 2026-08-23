"""Unit tests for Dewie API routes — using FastAPI TestClient."""

from __future__ import annotations

from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

# ── Search queue route ────────────────────────────────────────────────────────


def _make_search_queue_app(pg_mock=None):
    from dewie.api.routes.search_queue import router

    app = FastAPI()
    app.include_router(router)
    if pg_mock is None:
        pg_mock = AsyncMock()
    app.state.postgres = pg_mock
    return app


def test_search_queue_endpoint_exists():
    from dewie.api.routes.search_queue import router

    paths = [route.path for route in router.routes]
    assert any("/search-queue" in p or "search_queue" in p or "queue" in p for p in paths)


def test_search_queue_enqueue_success():
    pg = AsyncMock()
    pg.enqueue_search = AsyncMock(return_value=(True, "abc-123"))
    client = TestClient(_make_search_queue_app(pg))
    resp = client.post("/search-queue/enqueue", json={"query": "best cloud databases"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["queued"] is True
    assert data["id"] == "abc-123"


def test_search_queue_enqueue_duplicate_returns_false():
    pg = AsyncMock()
    pg.enqueue_search = AsyncMock(return_value=(False, None))
    client = TestClient(_make_search_queue_app(pg))
    resp = client.post("/search-queue/enqueue", json={"query": "duplicate query"})
    assert resp.status_code == 200
    assert resp.json()["queued"] is False


def test_search_queue_enqueue_empty_query_returns_422():
    client = TestClient(_make_search_queue_app())
    resp = client.post("/search-queue/enqueue", json={"query": ""})
    assert resp.status_code == 422


def test_search_queue_enqueue_query_too_long_returns_422():
    client = TestClient(_make_search_queue_app())
    resp = client.post("/search-queue/enqueue", json={"query": "x" * 501})
    assert resp.status_code == 422


def test_search_queue_enqueue_priority_out_of_range_returns_422():
    client = TestClient(_make_search_queue_app())
    resp = client.post("/search-queue/enqueue", json={"query": "test", "priority": 11})
    assert resp.status_code == 422


def test_search_queue_enqueue_category_optional():
    pg = AsyncMock()
    pg.enqueue_search = AsyncMock(return_value=(True, "xyz-456"))
    client = TestClient(_make_search_queue_app(pg))
    resp = client.post(
        "/search-queue/enqueue",
        json={"query": "test query", "category": "technology", "priority": 3},
    )
    assert resp.status_code == 200
    pg.enqueue_search.assert_called_once_with(query="test query", category="technology", priority=3)


# ── Capabilities route ────────────────────────────────────────────────────────


def test_capabilities_router_exists():
    from dewie.api.routes.capabilities import router

    assert router is not None
    assert len(router.routes) > 0


# ── Graph route ───────────────────────────────────────────────────────────────


def test_graph_router_exists():
    from dewie.api.routes.graph import router

    assert router is not None
    routes_paths = [r.path for r in router.routes]
    assert len(routes_paths) > 0


# ── Documents route ───────────────────────────────────────────────────────────


def test_documents_router_exists():
    from dewie.api.routes.documents import router

    assert router is not None


# ── Query route (basic import / structure) ────────────────────────────────────


def test_query_router_exists():
    from dewie.api.routes.query import router

    assert router is not None


# ── Traverse route ─────────────────────────────────────────────────────────────


def test_traverse_router_exists():
    from dewie.api.routes.traverse import router

    assert router is not None


# ── Admin route ─────────────────────────────────────────────────────────────────


def test_admin_router_exists():
    from dewie.api.routes.admin import router

    assert router is not None


# ── Corpus gap report route ───────────────────────────────────────────────────


def test_corpus_gap_report_router_exists():
    from dewie.api.routes.corpus import router

    assert router is not None
    paths = [r.path for r in router.routes]
    assert any("gap-report" in p for p in paths)


# ── /documents/ingest (issue #91) ─────────────────────────────────────────────


def _make_documents_app(pg_mock=None, processor_mock=None):
    """Build a minimal FastAPI app with just the documents router."""
    from dewie.api.routes.documents import router

    app = FastAPI()
    app.include_router(router)
    if pg_mock is None:
        pg_mock = AsyncMock()
        pg_mock.upsert = AsyncMock(return_value=None)
        pg_mock.write_body_text = AsyncMock(return_value=None)
    app.state.postgres = pg_mock
    app.state.processor = processor_mock  # None = enrichment disabled in tests
    return app


def test_documents_ingest_endpoint_exists():
    from dewie.api.routes.documents import router

    paths = [route.path for route in router.routes]
    assert any("ingest" in p for p in paths), f"No ingest route found in: {paths}"


def test_documents_ingest_raw_text_accepted():
    """POST /documents/ingest with raw text returns 202 and a doc_id."""
    pg = AsyncMock()
    pg.upsert = AsyncMock(return_value=None)
    pg.write_body_text = AsyncMock(return_value=None)
    app = _make_documents_app(pg_mock=pg)
    client = TestClient(app)

    resp = client.post(
        "/documents/ingest",
        json={"text": "This is a sample document for the corpus.", "title": "Test Doc"},
    )
    assert resp.status_code == 202, resp.text
    data = resp.json()
    assert "doc_id" in data
    assert data["status"] == "pending"
    assert "1 document" in data["message"]
    pg.upsert.assert_called_once()


def test_documents_ingest_url_and_text_uses_text():
    """When both url and text are provided, text is used directly (no fetch)."""
    pg = AsyncMock()
    pg.upsert = AsyncMock(return_value=None)
    pg.write_body_text = AsyncMock(return_value=None)
    app = _make_documents_app(pg_mock=pg)
    client = TestClient(app)

    resp = client.post(
        "/documents/ingest",
        json={
            "url": "https://example.com/article",
            "text": "Pre-fetched article body text.",
            "title": "Example Article",
            "corpus_id": "customer:acme",
        },
    )
    assert resp.status_code == 202, resp.text
    data = resp.json()
    assert "doc_id" in data


def test_documents_ingest_missing_url_and_text_returns_400():
    """POST /documents/ingest with neither url nor text returns 400."""
    app = _make_documents_app()
    client = TestClient(app)

    resp = client.post("/documents/ingest", json={"title": "No body"})
    assert resp.status_code == 400
    assert "url" in resp.json()["detail"] or "text" in resp.json()["detail"]


def test_documents_ingest_response_schema():
    """Response includes doc_id, status, and message fields."""
    pg = AsyncMock()
    pg.upsert = AsyncMock(return_value=None)
    pg.write_body_text = AsyncMock(return_value=None)
    app = _make_documents_app(pg_mock=pg)
    client = TestClient(app)

    resp = client.post("/documents/ingest", json={"text": "Hello world corpus document."})
    assert resp.status_code == 202
    data = resp.json()
    assert set(data.keys()) >= {"doc_id", "status", "message"}
    # doc_id should be a valid UUID-ish string
    import uuid

    uuid.UUID(data["doc_id"])  # raises if not valid


def test_documents_ingest_corpus_id_forwarded():
    """corpus_id from request is stored on the document."""

    pg = AsyncMock()
    pg.upsert = AsyncMock(return_value=None)
    pg.write_body_text = AsyncMock(return_value=None)
    app = _make_documents_app(pg_mock=pg)
    client = TestClient(app)

    client.post(
        "/documents/ingest",
        json={"text": "corpus doc", "corpus_id": "customer:test-corp"},
    )
    # upsert was called; verify the doc passed had corpus_id set
    assert pg.upsert.call_count == 1
    upserted_doc = pg.upsert.call_args[0][0]
    assert upserted_doc.corpus_id == "customer:test-corp"
