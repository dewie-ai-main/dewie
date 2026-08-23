"""Tests for dewie.api.routes.documents."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest


@pytest.mark.asyncio
async def test_inspect_document_never_exposes_answers_questions():
    from dewie.api.routes.documents import inspect_document

    doc_id = uuid4()
    row = {
        "id": doc_id,
        "url": "https://example.com",
        "title": "Example",
        "source": "web",
        "status": "ready",
        "retry_count": 0,
        "ingested_at": "2026-01-01T00:00:00Z",
        "enriched_at": "2026-01-01T00:01:00Z",
        "summary": "summary",
        "embed_summary": "embed",
        "topics": ["topic"],
        "keywords": ["keyword"],
        "entities": ["entity"],
        "answers_questions": ["q1", "q2"],
        "alternate_terms": ["alias"],
        "sentiment": "neutral",
        "tone": "informative",
        "document_type": "article",
        "reading_level": "general",
        "author": "A",
        "enrichment_quality_score": 0.9,
        "enrichment_version": "v1",
        "embedding_model": "text-embedding-3-small",
        "has_embedding": True,
        "embedding_dims": 1536,
    }

    doc_result = MagicMock()
    doc_result.mappings.return_value.one_or_none.return_value = row

    cache_result = MagicMock()
    cache_result.mappings.return_value.all.return_value = []

    session = AsyncMock()
    session.execute = AsyncMock(side_effect=[doc_result, cache_result])

    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(return_value=session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)

    pg = MagicMock()
    pg._is_sqlite = True
    pg._session_factory.return_value = session_ctx

    request = MagicMock()
    request.app.state.postgres = pg

    with patch("dewie.storage.body_store.load_body", return_value="body text"):
        result = await inspect_document(doc_id, request)

    assert "answers_questions" not in result
    assert result["aq_count"] == 2


@pytest.mark.asyncio
async def test_inspect_document_no_500_on_missing_optional_columns():
    """Issue #242 — inspect_document must return data (not 500) when the main SELECT
    query fails because optional columns (e.g. enrichment_quality_score, alternate_terms)
    are absent on older schema versions.  The fallback minimal query must succeed."""
    from dewie.api.routes.documents import inspect_document

    doc_id = uuid4()
    # Minimal row returned by the fallback query
    minimal_row = {
        "id": doc_id,
        "url": "https://example.com",
        "title": "Legacy doc",
        "source": "web",
        "status": "ready",
        "ingested_at": "2026-01-01T00:00:00Z",
        "enriched_at": None,
        "summary": "some summary",
        "topics": ["ai"],
        "keywords": ["ml"],
        "entities": [],
        "sentiment": 0.5,
    }

    # Full-query result mock raises (simulates missing column error)
    full_result = MagicMock()
    full_result.mappings.return_value.one_or_none.side_effect = Exception(
        'column "enrichment_quality_score" does not exist'
    )

    # Fallback minimal query result mock succeeds
    fallback_result = MagicMock()
    fallback_result.mappings.return_value.one_or_none.return_value = minimal_row

    # llm_cache query result (also succeeds)
    cache_result = MagicMock()
    cache_result.mappings.return_value.all.return_value = []

    call_count = 0

    async def execute_side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return full_result
        elif call_count == 2:
            return fallback_result
        return cache_result

    session = AsyncMock()
    session.execute = AsyncMock(side_effect=execute_side_effect)

    session_ctx = MagicMock()
    session_ctx.__aenter__ = AsyncMock(return_value=session)
    session_ctx.__aexit__ = AsyncMock(return_value=False)

    pg = MagicMock()
    pg._is_sqlite = True
    pg._session_factory.return_value = session_ctx

    request = MagicMock()
    request.app.state.postgres = pg

    with patch("dewie.storage.body_store.load_body", return_value=None):
        result = await inspect_document(doc_id, request)

    # Must return a valid dict, not raise
    assert isinstance(result, dict)
    assert result["title"] == "Legacy doc"
    assert result["url"] == "https://example.com"
    # Optional fields must have safe defaults
    assert result["embed_summary"] == ""
    assert result["alternate_terms"] == []
    assert result["aq_count"] == 0
    assert not result["has_embedding"]
    assert result["retry_count"] == 0
