"""Tests for dewie.api.routes.documents — route handlers."""

from __future__ import annotations

import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest


def _make_request(pg=None, cache=None, is_admin=False):
    req = MagicMock()
    req.app.state.postgres = pg or MagicMock()
    req.app.state.cache = cache or MagicMock()
    req.state.tenant_id = uuid.uuid4()
    req.state.is_admin = is_admin
    req.state.user_id = "user-1"
    return req


def _make_doc(doc_id=None):
    doc = MagicMock()
    doc.id = doc_id or uuid.uuid4()
    doc.title = "Test Document"
    doc.summary = "A test document"
    doc.url = "https://example.com/test"
    doc.source = "web"
    doc.topics = ["ai", "ml"]
    doc.keywords = ["python", "neural"]
    doc.entities = ["OpenAI"]
    doc.sentiment = 0.5
    doc.status = "READY"
    doc.ingested_at = datetime(2024, 1, 1)
    return doc


# ── get_document ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_document_not_found():
    from fastapi import HTTPException

    from dewie.api.routes.documents import get_document

    pg = MagicMock()
    pg.get_by_id = AsyncMock(return_value=None)
    req = _make_request(pg=pg)
    with pytest.raises(HTTPException) as exc:
        await get_document(uuid.uuid4(), req)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_get_document_success():
    from dewie.api.routes.documents import get_document

    pg = MagicMock()
    doc = _make_doc()
    pg.get_by_id = AsyncMock(return_value=doc)
    req = _make_request(pg=pg)
    result = await get_document(doc.id, req)
    assert result["title"] == "Test Document"
    assert result["url"] == "https://example.com/test"


# ── delete_document ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_document_requires_admin():
    from fastapi import HTTPException

    from dewie.api.routes.documents import delete_document

    req = _make_request(is_admin=False)
    with pytest.raises(HTTPException) as exc:
        await delete_document(uuid.uuid4(), req)
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_delete_document_not_found():
    from fastapi import HTTPException

    from dewie.api.routes.documents import delete_document

    pg = MagicMock()
    pg.get_by_id = AsyncMock(return_value=None)
    req = _make_request(pg=pg, is_admin=True)
    with pytest.raises(HTTPException) as exc:
        await delete_document(uuid.uuid4(), req)
    assert exc.value.status_code == 404


@pytest.mark.asyncio
async def test_delete_document_success():
    from dewie.api.routes.documents import delete_document

    pg = MagicMock()
    doc = _make_doc()
    pg.get_by_id = AsyncMock(return_value=doc)

    conn = AsyncMock()
    conn.execute = AsyncMock()
    begin_cm = AsyncMock()
    begin_cm.__aenter__ = AsyncMock(return_value=conn)
    begin_cm.__aexit__ = AsyncMock(return_value=False)
    pg._engine.begin.return_value = begin_cm

    cache = MagicMock()
    cache._redis.delete = AsyncMock()
    req = _make_request(pg=pg, cache=cache, is_admin=True)

    from unittest.mock import patch

    with (
        patch("dewie.storage.body_store.delete_body"),
        patch("asyncio.to_thread", AsyncMock(return_value=None)),
    ):
        await delete_document(doc.id, req)  # should not raise

    delete_calls = [
        c for c in conn.execute.call_args_list if "DELETE FROM documents" in str(c.args[0])
    ]
    assert len(delete_calls) == 1, "expected exactly one document DELETE"


# ── list_documents ────────────────────────────────────────────────────────────


def _make_doc_list_app(pg_mock, cache_mock=None):
    from fastapi import FastAPI

    from dewie.api.middleware import limiter
    from dewie.api.routes.documents import router as docs_router

    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(docs_router)
    app.state.postgres = pg_mock
    app.state.cache = cache_mock or MagicMock()
    return app


def test_list_documents_returns_list():
    from fastapi.testclient import TestClient

    doc = _make_doc()
    doc.status = MagicMock()
    doc.status.value = "ready"
    pg = AsyncMock()
    pg.list_recent = AsyncMock(return_value=[doc])
    app = _make_doc_list_app(pg)
    client = TestClient(app)
    resp = client.get("/documents/list")
    assert resp.status_code == 200
    body = resp.json()
    assert "documents" in body
    assert len(body["documents"]) == 1
    assert body["documents"][0]["title"] == "Test Document"


def test_list_documents_empty():
    from fastapi.testclient import TestClient

    pg = AsyncMock()
    pg.list_recent = AsyncMock(return_value=[])
    app = _make_doc_list_app(pg)
    client = TestClient(app)
    resp = client.get("/documents/list")
    assert resp.status_code == 200
    assert resp.json()["documents"] == []


# ── get_content ───────────────────────────────────────────────────────────────


def test_get_content_not_found():
    from fastapi.testclient import TestClient

    pg = AsyncMock()
    pg.get_by_id = AsyncMock(return_value=None)
    cache = MagicMock()
    cache._redis = AsyncMock()
    cache._redis.get = AsyncMock(return_value=None)
    app = _make_doc_list_app(pg, cache_mock=cache)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get(f"/documents/{uuid.uuid4()}/content")
    assert resp.status_code == 404


def test_get_content_postgres_body_text():
    """get_content reads body_text from Postgres (issue #119 — Redis removed from body read path)."""
    from unittest.mock import MagicMock as _MagicMock

    from fastapi.testclient import TestClient

    doc_id = uuid.uuid4()
    doc = _make_doc(doc_id=doc_id)
    pg = AsyncMock()
    pg.get_by_id = AsyncMock(return_value=doc)

    # Mock the session to return body_text from Postgres
    row = _MagicMock()
    row.__getitem__ = lambda self, k: "postgres body content" if k == "body_text" else None
    session_result = _MagicMock()
    session_result.mappings.return_value.first.return_value = row
    session = AsyncMock()
    session.execute = AsyncMock(return_value=session_result)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    session_factory = _MagicMock(return_value=session)
    pg._session_factory = session_factory

    app = _make_doc_list_app(pg)
    client = TestClient(app)
    resp = client.get(f"/documents/{doc_id}/content")
    assert resp.status_code == 200
    assert resp.text == "postgres body content"


def test_get_content_file_fallback():
    """get_content falls back to flat-file body store when Postgres body_text is absent."""
    from unittest.mock import MagicMock as _MagicMock
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    doc_id = uuid.uuid4()
    doc = _make_doc(doc_id=doc_id)
    pg = AsyncMock()
    pg.get_by_id = AsyncMock(return_value=doc)

    # Postgres returns no body_text
    row = _MagicMock()
    row.__getitem__ = lambda self, k: None
    session_result = _MagicMock()
    session_result.mappings.return_value.first.return_value = row
    session = AsyncMock()
    session.execute = AsyncMock(return_value=session_result)
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    pg._session_factory = _MagicMock(return_value=session)

    app = _make_doc_list_app(pg)
    client = TestClient(app)
    with patch("dewie.storage.body_store.load_body", return_value="file body content"):
        resp = client.get(f"/documents/{doc_id}/content")
    assert resp.status_code == 200
    assert resp.text == "file body content"


# ── get_chunks ────────────────────────────────────────────────────────────────


def test_get_chunks_not_found():
    from fastapi.testclient import TestClient

    pg = AsyncMock()
    pg.get_by_id = AsyncMock(return_value=None)
    app = _make_doc_list_app(pg)
    client = TestClient(app, raise_server_exceptions=False)
    resp = client.get(f"/documents/{uuid.uuid4()}/chunks")
    assert resp.status_code == 404


def test_get_chunks_success():
    from fastapi.testclient import TestClient

    doc_id = uuid.uuid4()
    doc = _make_doc(doc_id=doc_id)
    chunks = [{"chunk_index": 0, "text": "first chunk"}, {"chunk_index": 1, "text": "second chunk"}]
    pg = AsyncMock()
    pg.get_by_id = AsyncMock(return_value=doc)
    pg.get_chunks = AsyncMock(return_value=chunks)
    app = _make_doc_list_app(pg)
    client = TestClient(app)
    resp = client.get(f"/documents/{doc_id}/chunks")
    assert resp.status_code == 200
    body = resp.json()
    assert body["chunk_count"] == 2
    assert body["doc_id"] == str(doc_id)
    assert body["chunks"] == chunks
