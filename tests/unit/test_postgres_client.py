"""Tests for dewie.storage.postgres — pure helpers and mocked client methods."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest

# ── _safe_enum ────────────────────────────────────────────────────────────────


def test_safe_enum_valid():
    from dewie.models.content import ContentStatus
    from dewie.storage.postgres import _safe_enum

    assert _safe_enum(ContentStatus, "ready") == ContentStatus.READY


def test_safe_enum_invalid_returns_none():
    from dewie.models.content import ContentStatus
    from dewie.storage.postgres import _safe_enum

    assert _safe_enum(ContentStatus, "nonexistent") is None


def test_safe_enum_none_returns_none():
    from dewie.models.content import ContentStatus
    from dewie.storage.postgres import _safe_enum

    assert _safe_enum(ContentStatus, None) is None


# ── _row_to_doc ───────────────────────────────────────────────────────────────


def _make_row(**overrides):
    base = {
        "id": uuid.uuid4(),
        "url": "https://example.com/article",
        "title": "Test Title",
        "summary": "Test summary",
        "source": "web",
        "ingested_at": datetime.now(UTC),
        "status": "ready",
        "topics": ["tech", "ai"],
        "keywords": ["python"],
        "entities": ["OpenAI"],
        "sentiment": 0.5,
        "crawl_session": None,
        "enrichment_version": 1,
        "embedding_model": "text-embedding-3-small",
        "enriched_at": datetime.now(UTC),
        "answers_questions": ["What is this?"],
        "tone": "neutral",
        "document_type": None,
        "author": "Alice",
        "reading_level": None,
        "embed_summary": "embed text",
        "published_at": None,
        "paywall_detected": False,
        "paywall_type": "none",
        "alternate_terms": [],
        "enrichment_quality_score": 75,
        "gap_fill": False,
        "corpus_id": None,
        "language": "en",
    }
    base.update(overrides)
    return base


def test_row_to_doc_basic():
    from dewie.storage.postgres import _row_to_doc

    row = _make_row()
    doc = _row_to_doc(row)
    assert doc.url == "https://example.com/article"
    assert doc.title == "Test Title"
    assert doc.source == "web"
    assert doc.enrichment_quality_score == 75


def test_row_to_doc_json_string_topics():
    from dewie.storage.postgres import _row_to_doc

    row = _make_row(topics=json.dumps(["ai", "ml"]))
    doc = _row_to_doc(row)
    assert doc.topics == ["ai", "ml"]


def test_row_to_doc_null_optional_fields():
    from dewie.storage.postgres import _row_to_doc

    row = _make_row(
        answers_questions=None,
        alternate_terms=None,
        embed_summary=None,
        paywall_detected=None,
        gap_fill=None,
    )
    doc = _row_to_doc(row)
    assert doc.answers_questions == []
    assert doc.alternate_terms == []
    assert doc.embed_summary == ""
    assert doc.paywall_detected is False
    assert doc.gap_fill is False


def test_row_to_doc_valid_document_type():
    from dewie.models.content import DocumentType
    from dewie.storage.postgres import _row_to_doc

    row = _make_row(document_type="blog_post")
    doc = _row_to_doc(row)
    assert doc.document_type == DocumentType.BLOG_POST


def test_row_to_doc_invalid_document_type_is_none():
    from dewie.storage.postgres import _row_to_doc

    row = _make_row(document_type="unknown_type_xyz")
    doc = _row_to_doc(row)
    assert doc.document_type is None


def test_row_to_doc_unknown_status_defaults_to_pending():
    """Issue #242 — legacy 'deferred' status must not raise ValueError → 500."""
    from dewie.models.content import ContentStatus
    from dewie.storage.postgres import _row_to_doc

    row = _make_row(status="deferred")
    doc = _row_to_doc(row)
    assert doc.status == ContentStatus.PENDING


def test_row_to_doc_known_statuses_preserved():
    """All known ContentStatus values must round-trip correctly through _row_to_doc."""
    from dewie.models.content import ContentStatus
    from dewie.storage.postgres import _row_to_doc

    for s in ContentStatus:
        row = _make_row(status=s.value)
        doc = _row_to_doc(row)
        assert doc.status == s


# ── PostgresClient methods (mocked engine) ────────────────────────────────────


def _make_pg():
    from dewie.storage.postgres import PostgresClient

    pg = object.__new__(PostgresClient)
    pg._engine = MagicMock()
    pg._session_factory = MagicMock()
    return pg


def _mock_session_cm(rows=None, scalar=None):
    session = AsyncMock()
    result = MagicMock()
    if rows is not None:
        result.mappings.return_value.all.return_value = rows
        result.mappings.return_value.first.return_value = rows[0] if rows else None
    if scalar is not None:
        result.scalar.return_value = scalar
    session.execute = AsyncMock(return_value=result)
    session.commit = AsyncMock()
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm, session


def _mock_conn_cm(rows=None, scalar=None):
    conn = AsyncMock()
    result = MagicMock()
    if rows is not None:
        result.mappings.return_value.all.return_value = rows
        result.fetchall.return_value = rows
        result.fetchone.return_value = rows[0] if rows else None
    if scalar is not None:
        result.scalar.return_value = scalar
    conn.execute = AsyncMock(return_value=result)
    conn.exec_driver_sql = AsyncMock(return_value=result)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm, conn


@pytest.mark.asyncio
async def test_get_pending_docs_uses_atomic_cte_on_postgres():
    """Postgres path must use _engine.begin() (FOR UPDATE SKIP LOCKED), not _session_factory.

    A previous fix accidentally introduced a _session_factory short-circuit that bypassed
    the atomic CTE claim, allowing multiple concurrent workers to pick the same doc.
    This test locks that path down.
    """
    pg = _make_pg()
    rows = [{"id": str(uuid.uuid4())}, {"id": str(uuid.uuid4())}]
    cm, conn = _mock_conn_cm(rows=rows)
    pg._engine.begin.return_value = cm
    result = await pg.get_pending_docs(limit=2)
    # Must have gone through _engine.begin(), NOT _session_factory
    pg._engine.begin.assert_called_once()
    pg._session_factory.assert_not_called()
    assert len(result) == 2
    assert all(isinstance(r, str) for r in result)


@pytest.mark.asyncio
async def test_get_pending_docs_sqlite_uses_session_factory():
    """SQLite path must use _session_factory (no FOR UPDATE SKIP LOCKED support)."""
    pg = _make_pg()
    pg._is_sqlite = True
    rows = [{"id": str(uuid.uuid4())}, {"id": str(uuid.uuid4())}]
    cm, _ = _mock_session_cm(rows=rows)
    pg._session_factory.return_value = cm
    result = await pg.get_pending_docs(limit=2)
    # Must have gone through _session_factory, NOT _engine.begin()
    pg._session_factory.assert_called_once()
    pg._engine.begin.assert_not_called()
    assert len(result) == 2
    assert all(isinstance(r, str) for r in result)


@pytest.mark.asyncio
async def test_count_by_status_returns_dict():
    pg = _make_pg()
    rows = [{"status": "ready", "n": 10}, {"status": "pending", "n": 3}]
    cm, _ = _mock_session_cm(rows=rows)
    pg._session_factory.return_value = cm
    result = await pg.count_by_status()
    assert result == {"ready": 10, "pending": 3}


@pytest.mark.asyncio
async def test_mark_status_calls_commit():
    pg = _make_pg()
    cm, session = _mock_session_cm()
    pg._session_factory.return_value = cm
    from dewie.models.content import ContentStatus

    doc_id = uuid.uuid4()
    await pg.mark_status(doc_id, ContentStatus.READY)
    session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_get_by_id_returns_none_when_missing():
    pg = _make_pg()
    session = AsyncMock()
    result = MagicMock()
    result.mappings.return_value.first.return_value = None
    session.execute = AsyncMock(return_value=result)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=False)
    pg._session_factory.return_value = cm

    result = await pg.get_by_id(uuid.uuid4())
    assert result is None


@pytest.mark.asyncio
async def test_set_embedding_calls_commit():
    pg = _make_pg()
    cm, session = _mock_session_cm()
    pg._session_factory.return_value = cm
    doc_id = uuid.uuid4()
    await pg.set_embedding(doc_id, [0.1, 0.2, 0.3])
    session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_write_body_text_calls_commit():
    pg = _make_pg()
    cm, session = _mock_session_cm()
    pg._session_factory.return_value = cm
    await pg.write_body_text(uuid.uuid4(), "some body text")
    session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_search_queue_depth_returns_count():
    pg = _make_pg()
    conn = AsyncMock()
    result = MagicMock()
    result.first.return_value = (42,)
    conn.execute = AsyncMock(return_value=result)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    pg._engine.connect.return_value = cm

    depth = await pg.search_queue_depth()
    assert depth == 42


@pytest.mark.asyncio
async def test_review_queue_depth_returns_count():
    pg = _make_pg()
    conn = AsyncMock()
    result = MagicMock()
    result.scalar.return_value = 3
    conn.execute = AsyncMock(return_value=result)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    pg._engine.connect.return_value = cm

    depth = await pg.review_queue_depth()
    assert depth == 3


@pytest.mark.asyncio
async def test_find_by_topics_empty_returns_empty():
    pg = _make_pg()
    result = await pg.find_by_topics([])
    assert result == []


@pytest.mark.asyncio
async def test_find_by_entities_empty_returns_empty():
    pg = _make_pg()
    result = await pg.find_by_entities([])
    assert result == []


@pytest.mark.asyncio
async def test_find_by_keywords_empty_returns_empty():
    pg = _make_pg()
    result = await pg.find_by_keywords([])
    assert result == []


@pytest.mark.asyncio
async def test_search_chunks_for_docs_empty_returns_empty():
    pg = _make_pg()
    result = await pg.search_chunks_for_docs("query", [])
    assert result == {}


@pytest.mark.asyncio
async def test_close_disposes_engine():
    pg = _make_pg()
    pg._engine.dispose = AsyncMock()
    await pg.close()
    pg._engine.dispose.assert_called_once()


@pytest.mark.skip(reason="get_waitlist_count removed in Dewie rename")
@pytest.mark.asyncio
async def test_get_waitlist_count():
    pass


@pytest.mark.asyncio
async def test_get_edge_count():
    pg = _make_pg()
    session = AsyncMock()
    result = MagicMock()
    result.scalar.return_value = 5
    session.execute = AsyncMock(return_value=result)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=session)
    cm.__aexit__ = AsyncMock(return_value=False)
    pg._session_factory.return_value = cm

    count = await pg.get_edge_count(uuid.uuid4())
    assert count == 5


@pytest.mark.asyncio
async def test_upsert_aq_embeddings_noop_on_empty():
    pg = _make_pg()
    await pg.upsert_aq_embeddings("doc-id", [])
    pg._engine.begin.assert_not_called()


@pytest.mark.asyncio
async def test_insert_chunks_noop_on_empty():
    pg = _make_pg()
    await pg.insert_chunks(uuid.uuid4(), [])
    pg._engine.begin.assert_not_called()


# ── User CRUD methods ─────────────────────────────────────────────────────────


def _make_row_mock(mapping: dict):
    """Create a row mock that supports _mapping."""
    row = MagicMock()
    row._mapping = mapping
    return row


def _mock_engine_connect_cm(fetchone_return=None, fetchall_return=None):
    conn = AsyncMock()
    result = MagicMock()
    result.fetchone.return_value = fetchone_return  # explicitly set (even if None)
    if fetchall_return is not None:
        result.fetchall.return_value = fetchall_return
    conn.execute = AsyncMock(return_value=result)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm, conn


def _mock_engine_begin_cm(fetchone_return=None, fetchall_return=None, rowcount=1):
    conn = AsyncMock()
    result = MagicMock()
    result.fetchone.return_value = fetchone_return  # explicitly set (even if None)
    if fetchall_return is not None:
        result.fetchall.return_value = fetchall_return
    result.rowcount = rowcount
    conn.execute = AsyncMock(return_value=result)
    conn.exec_driver_sql = AsyncMock(return_value=result)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm, conn


# ── Workspace / corpus CRUD ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_workspaces_returns_list():
    pg = _make_pg()
    row = MagicMock()
    row._mapping = {"id": str(uuid.uuid4()), "name": "default", "parent_id": None}
    cm, _ = _mock_engine_connect_cm(fetchall_return=[row])
    pg._engine.connect.return_value = cm
    result = await pg.get_workspaces()
    assert len(result) == 1


@pytest.mark.asyncio
async def test_get_workspaces_with_parent_id_filters():
    pg = _make_pg()
    parent = uuid.uuid4()
    cm, conn = _mock_engine_connect_cm(fetchall_return=[])
    pg._engine.connect.return_value = cm
    result = await pg.get_workspaces(parent_id=parent)
    assert result == []
    conn.execute.assert_called_once()


@pytest.mark.asyncio
async def test_create_workspace_returns_dict():
    pg = _make_pg()
    row = MagicMock()
    row._mapping = {"id": str(uuid.uuid4()), "name": "test-ws", "parent_id": None, "sharing_tier": "internal_only", "created_at": "2025-01-01"}
    cm, _ = _mock_engine_begin_cm(fetchone_return=row)
    pg._engine.begin.return_value = cm
    result = await pg.create_workspace("test-ws")
    assert result["name"] == "test-ws"


@pytest.mark.asyncio
async def test_create_workspace_with_parent_id():
    pg = _make_pg()
    parent_id = uuid.uuid4()
    row = MagicMock()
    row._mapping = {"id": str(uuid.uuid4()), "name": "child-ws", "parent_id": str(parent_id), "sharing_tier": "internal_only", "created_at": "2025-01-01"}
    cm, conn = _mock_engine_begin_cm(fetchone_return=row)
    pg._engine.begin.return_value = cm
    result = await pg.create_workspace("child-ws", parent_id=parent_id)
    assert result is not None
    conn.execute.assert_called_once()


@pytest.mark.asyncio
async def test_delete_workspace_calls_execute():
    pg = _make_pg()
    cm, conn = _mock_engine_begin_cm()
    pg._engine.begin.return_value = cm
    ws_id = uuid.uuid4()
    await pg.delete_workspace(ws_id)
    conn.execute.assert_called_once()


@pytest.mark.asyncio
async def test_get_corpora_returns_list():
    pg = _make_pg()
    row = MagicMock()
    row._mapping = {"id": str(uuid.uuid4()), "name": "default", "slug": "default", "workspace_id": str(uuid.uuid4())}
    cm, _ = _mock_engine_connect_cm(fetchall_return=[row])
    pg._engine.connect.return_value = cm
    result = await pg.get_corpora()
    assert len(result) == 1


@pytest.mark.asyncio
async def test_get_corpora_with_workspace_id_filters():
    pg = _make_pg()
    ws_id = uuid.uuid4()
    cm, conn = _mock_engine_connect_cm(fetchall_return=[])
    pg._engine.connect.return_value = cm
    result = await pg.get_corpora(workspace_id=ws_id)
    assert result == []
    conn.execute.assert_called_once()


@pytest.mark.asyncio
async def test_create_corpus_returns_dict():
    pg = _make_pg()
    ws_id = uuid.uuid4()
    row = MagicMock()
    row._mapping = {"id": str(uuid.uuid4()), "name": "Main", "slug": "main", "workspace_id": str(ws_id), "sharing_tier": "internal_only", "created_at": "2025-01-01"}
    cm, _ = _mock_engine_begin_cm(fetchone_return=row)
    pg._engine.begin.return_value = cm
    result = await pg.create_corpus(name="Main", slug="main", workspace_id=ws_id)
    assert result["slug"] == "main"


@pytest.mark.asyncio
async def test_delete_corpus_calls_execute():
    pg = _make_pg()
    cm, conn = _mock_engine_begin_cm()
    pg._engine.begin.return_value = cm
    corpus_id = uuid.uuid4()
    await pg.delete_corpus(corpus_id)
    conn.execute.assert_called_once()


@pytest.mark.asyncio
async def test_get_or_create_default_corpus_returns_existing():
    from dewie.storage.postgres import DEFAULT_CORPUS_ID

    pg = _make_pg()
    row = MagicMock()
    row._mapping = {"id": str(DEFAULT_CORPUS_ID), "name": "Default", "slug": "default", "workspace_id": str(uuid.uuid4()), "sharing_tier": "internal_only", "created_at": "2025-01-01"}
    cm, _ = _mock_engine_connect_cm(fetchone_return=row)
    pg._engine.connect.return_value = cm
    result = await pg.get_or_create_default_corpus()
    assert result is not None


@pytest.mark.asyncio
async def test_get_or_create_default_corpus_creates_when_missing():
    from dewie.storage.postgres import DEFAULT_CORPUS_ID, ROOT_WORKSPACE_ID

    pg = _make_pg()
    # connect returns None (not found)
    conn_cm, _ = _mock_engine_connect_cm(fetchone_return=None)
    pg._engine.connect.return_value = conn_cm
    # begin returns new row
    row = MagicMock()
    row._mapping = {"id": str(DEFAULT_CORPUS_ID), "name": "Default", "slug": "default", "workspace_id": str(ROOT_WORKSPACE_ID), "sharing_tier": "internal_only", "created_at": "2025-01-01"}
    begin_cm, _ = _mock_engine_begin_cm(fetchone_return=row)
    pg._engine.begin.return_value = begin_cm
    result = await pg.get_or_create_default_corpus()
    assert result is not None


# ── Document find methods ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_find_by_topics_empty_returns_empty_v2():
    pg = _make_pg()
    result = await pg.find_by_topics([])
    assert result == []


@pytest.mark.asyncio
async def test_find_by_topics_returns_docs():
    pg = _make_pg()
    row = dict(_make_row())
    cm, _ = _mock_session_cm(rows=[row])
    pg._session_factory.return_value = cm
    docs = await pg.find_by_topics(["ai"])
    assert len(docs) == 1


@pytest.mark.asyncio
async def test_find_by_entities_empty_returns_empty_v2():
    pg = _make_pg()
    result = await pg.find_by_entities([])
    assert result == []


@pytest.mark.asyncio
async def test_find_by_entities_returns_docs():
    pg = _make_pg()
    row = dict(_make_row())
    cm, _ = _mock_session_cm(rows=[row])
    pg._session_factory.return_value = cm
    docs = await pg.find_by_entities(["OpenAI"])
    assert len(docs) == 1


@pytest.mark.asyncio
async def test_find_by_keywords_empty_returns_empty_v2():
    pg = _make_pg()
    result = await pg.find_by_keywords([])
    assert result == []


@pytest.mark.asyncio
async def test_find_by_keywords_returns_docs():
    pg = _make_pg()
    row = dict(_make_row())
    cm, _ = _mock_session_cm(rows=[row])
    pg._session_factory.return_value = cm
    docs = await pg.find_by_keywords(["python"])
    assert len(docs) == 1


@pytest.mark.asyncio
async def test_list_recent_returns_docs():
    pg = _make_pg()
    row = dict(_make_row())
    cm, _ = _mock_session_cm(rows=[row])
    pg._session_factory.return_value = cm
    docs = await pg.list_recent(limit=10)
    assert len(docs) == 1


@pytest.mark.asyncio
async def test_list_crawl_sessions_returns_list():
    pg = _make_pg()
    row = {
        "crawl_session": "sess1",
        "total": 5,
        "ready": 4,
        "processing": 0,
        "failed": 1,
        "started_at": None,
        "last_seen_at": None,
    }
    cm, session = _mock_session_cm(rows=[row])
    pg._session_factory.return_value = cm
    sessions = await pg.list_crawl_sessions()
    assert len(sessions) == 1


# ── Relationships / edges ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_upsert_relationship_calls_execute():
    from dewie.models.metadata import Relationship, RelationshipType

    pg = _make_pg()
    cm, conn = _mock_engine_begin_cm()
    pg._engine.begin.return_value = cm
    rel = Relationship(
        source_id=uuid.uuid4(),
        target_id=uuid.uuid4(),
        relationship_type=RelationshipType.SHARED_TOPIC,
        weight=0.8,
        shared_attributes=[],
    )
    await pg.upsert_relationship(rel)
    conn.execute.assert_called_once()


@pytest.mark.asyncio
async def test_get_related_returns_list():
    pg = _make_pg()
    row = {
        "id": str(uuid.uuid4()),
        "rel_type": "related",
        "weight": 0.9,
        "shared": ["ai"],
        "title": "Test",
        "summary": "A test",
        "topics": ["tech"],
    }
    cm, _ = _mock_session_cm(rows=[row])
    pg._session_factory.return_value = cm
    results = await pg.get_related(uuid.uuid4(), ["related"], limit=5)
    assert len(results) == 1
    assert results[0]["rel_type"] == "related"


# ── Embedding and body ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_set_embedding_commits():
    pg = _make_pg()
    cm, session = _mock_session_cm()
    pg._session_factory.return_value = cm
    await pg.set_embedding(uuid.uuid4(), [0.1, 0.2, 0.3])
    session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_write_body_text_commits():
    pg = _make_pg()
    cm, session = _mock_session_cm()
    pg._session_factory.return_value = cm
    await pg.write_body_text(uuid.uuid4(), "Body text here")
    session.commit.assert_called_once()


@pytest.mark.asyncio
async def test_set_priority_commits():
    pg = _make_pg()
    cm, session = _mock_session_cm()
    pg._session_factory.return_value = cm
    await pg.set_priority(uuid.uuid4(), 10)
    session.commit.assert_called_once()


# ── Chunks ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_chunks_returns_list():
    pg = _make_pg()
    row = MagicMock()
    row.__getitem__ = lambda self, k: {"chunk_index": 0, "text": "chunk text"}[k]
    row._mapping = {"chunk_index": 0, "text": "chunk text"}

    conn = AsyncMock()
    result = MagicMock()
    result.mappings.return_value.all.return_value = [{"chunk_index": 0, "text": "chunk text"}]
    conn.execute = AsyncMock(return_value=result)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    pg._engine.connect.return_value = cm

    chunks = await pg.get_chunks(uuid.uuid4())
    assert len(chunks) == 1
    assert chunks[0]["text"] == "chunk text"


@pytest.mark.asyncio
async def test_insert_chunks_with_data():
    pg = _make_pg()
    cm, conn = _mock_engine_begin_cm()
    pg._engine.begin.return_value = cm
    doc_id = uuid.uuid4()
    await pg.insert_chunks(doc_id, [(0, "chunk text", [0.1, 0.2])])
    assert conn.execute.call_count >= 2  # DELETE + INSERT


@pytest.mark.asyncio
async def test_mark_chunk_status_calls_execute():
    pg = _make_pg()
    cm, conn = _mock_engine_begin_cm()
    pg._engine.begin.return_value = cm
    await pg.mark_chunk_status(uuid.uuid4(), "chunked")
    conn.execute.assert_called_once()


@pytest.mark.asyncio
async def test_get_unchunked_docs_returns_list():
    pg = _make_pg()
    conn = AsyncMock()
    result = MagicMock()
    result.fetchall.return_value = [("id1", "Title", "web", "body text")]
    conn.execute = AsyncMock(return_value=result)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    pg._engine.connect.return_value = cm
    docs = await pg.get_unchunked_docs(limit=10)
    assert docs == [("id1", "Title", "web", "body text")]


# ── Search queue ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_enqueue_search_new_item():
    pg = _make_pg()
    cm, conn = _mock_engine_begin_cm()
    # First execute (dedup check) returns None
    first_result = MagicMock()
    first_result.first.return_value = None
    # Second execute (insert) returns new id
    second_result = MagicMock()
    second_result.first.return_value = ("sq-id-1",)

    call_count = [0]

    async def fake_execute(sql, params=None):
        call_count[0] += 1
        return first_result if call_count[0] == 1 else second_result

    conn.execute = fake_execute

    pg._engine.begin.return_value = cm
    queued, item_id = await pg.enqueue_search("test query")
    assert queued is True
    assert item_id == "sq-id-1"


@pytest.mark.asyncio
async def test_enqueue_search_duplicate_returns_false():
    pg = _make_pg()
    cm, conn = _mock_engine_begin_cm()
    existing_result = MagicMock()
    existing_result.first.return_value = ("existing-id",)
    conn.execute = AsyncMock(return_value=existing_result)
    pg._engine.begin.return_value = cm
    queued, item_id = await pg.enqueue_search("duplicate query")
    assert queued is False
    assert item_id is None


@pytest.mark.asyncio
async def test_dequeue_search_batch_returns_rows():
    pg = _make_pg()
    row = {"id": "sq1", "query": "test", "category": None, "priority": 5}
    cm, conn = _mock_engine_begin_cm()
    result = MagicMock()
    result.mappings.return_value.all.return_value = [row]
    conn.execute = AsyncMock(return_value=result)
    pg._engine.begin.return_value = cm
    batch = await pg.dequeue_search_batch(batch_size=5)
    assert len(batch) == 1


@pytest.mark.asyncio
async def test_mark_search_queue_status_calls_execute():
    pg = _make_pg()
    cm, conn = _mock_engine_begin_cm()
    pg._engine.begin.return_value = cm
    await pg.mark_search_queue_status("sq-id-1", "done")
    conn.execute.assert_called_once()


@pytest.mark.asyncio
async def test_search_queue_depth_returns_int():
    pg = _make_pg()
    cm, conn = _mock_engine_connect_cm()
    row = MagicMock()
    row.__getitem__ = lambda self, k: 7
    result = MagicMock()
    result.first.return_value = row
    conn.execute = AsyncMock(return_value=result)
    pg._engine.connect.return_value = cm
    depth = await pg.search_queue_depth()
    assert depth == 7


@pytest.mark.asyncio
async def test_add_to_review_queue_calls_execute():
    pg = _make_pg()
    cm, conn = _mock_engine_begin_cm()
    pg._engine.begin.return_value = cm
    await pg.add_to_review_queue(uuid.uuid4(), "failed", "https://example.com")
    conn.execute.assert_called_once()


@pytest.mark.asyncio
async def test_review_queue_depth_returns_int():
    pg = _make_pg()
    conn = AsyncMock()
    result = MagicMock()
    result.scalar.return_value = 3
    conn.execute = AsyncMock(return_value=result)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=conn)
    cm.__aexit__ = AsyncMock(return_value=False)
    pg._engine.connect.return_value = cm
    depth = await pg.review_queue_depth()
    assert depth == 3


# ── schema migration completeness ─────────────────────────────────────────────


def _read_all_migration_sources() -> str:
    """Concatenated source of every Alembic revision file."""
    from pathlib import Path

    import dewie

    versions = Path(dewie.__file__).parent / "migrations" / "versions"
    return "\n".join(p.read_text(encoding="utf-8") for p in sorted(versions.glob("*.py")))


def test_document_chunks_aq_columns_in_migrations():
    """aq_text and aq_embedding must appear in the incremental migration list.

    These columns were prod-only drift (closes #175). This test ensures they
    stay in the OSS migration list and can't be accidentally dropped again.
    """
    # Schema moved from inline _SCHEMA_STATEMENTS to Alembic revision files;
    # the guarantee (aq columns + HNSW index present) is asserted against those.
    migration_sql = _read_all_migration_sources()
    assert "document_chunks" in migration_sql and "aq_text" in migration_sql, (
        "aq_text column migration missing from document_chunks"
    )
    assert "document_chunks" in migration_sql and "aq_embedding" in migration_sql, (
        "aq_embedding column migration missing from document_chunks"
    )
    # HNSW index on aq_embedding must also be present (name matches prod)
    assert "document_chunks_aq_embedding_idx" in migration_sql, (
        "HNSW index on document_chunks.aq_embedding missing from migrations"
    )


def test_document_chunks_aq_columns_in_sqlite_schema():
    """aq_text and aq_embedding must be in the SQLite CREATE TABLE statement.

    SQLite doesn't support ADD COLUMN with vector types — the columns must
    be in the initial CREATE TABLE definition.
    """
    import inspect

    import dewie.storage.postgres as pg_module

    # Use inspect to get just the _init_sqlite_schema function source — avoids
    # matching the call-site reference in init_schema().
    src = inspect.getsource(pg_module.PostgresClient._init_sqlite_schema)
    # Find the document_chunks CREATE TABLE block inside it
    chunk_start = src.find("document_chunks")
    assert chunk_start != -1, "document_chunks not found in _init_sqlite_schema"
    chunk_block = src[chunk_start:chunk_start + 600]
    assert "aq_text" in chunk_block, "aq_text missing from SQLite document_chunks schema"
    assert "aq_embedding" in chunk_block, "aq_embedding missing from SQLite document_chunks schema"


def test_issue_244_dev_user_seeded_without_google_sub():
    """Issue #244: /auth/me must never report auth_method='google' for the dev user.

    Originally guarded an UPDATE clearing google_sub in the inline migration
    list. Under Alembic the dev user is seeded fresh, so the invariant is now:
    the seed INSERT must not set google_sub at all.
    """
    migration_sql = _read_all_migration_sources()
    dev_user_id = "00000000-0000-0000-0000-000000000002"
    assert dev_user_id in migration_sql, "dev user seed missing from migrations"

    seed_start = migration_sql.index(dev_user_id)
    seed_block = migration_sql[seed_start - 400 : seed_start + 400]
    insert_block = [b for b in seed_block.split("op.execute") if dev_user_id in b and "INSERT INTO users" in b]
    assert insert_block, "dev user INSERT not found near its UUID"
    assert "google_sub" not in insert_block[0], (
        "dev user seed must not set google_sub (auth_method would report 'google')"
    )
