"""Tests for dewie.enrichment.chunk_embedder."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest

from dewie.enrichment.chunk_embedder import _domain, embed_and_store_chunks, run_once

# ── _domain helper ────────────────────────────────────────────────────────────


def test_domain_extracts_hostname():
    assert _domain("https://example.com/path") == "example.com"


def test_domain_handles_plain_string():
    assert _domain("youtube") == "youtube"


def test_domain_handles_empty():
    assert _domain("") == ""


# ── embed_and_store_chunks ────────────────────────────────────────────────────


def _make_pg():
    pg = AsyncMock()
    pg.mark_chunk_status = AsyncMock()
    pg.insert_chunks = AsyncMock()
    return pg


def _make_provider(vectors=None):
    provider = AsyncMock()
    provider.embed = AsyncMock(return_value=vectors)
    return provider


@pytest.mark.asyncio
async def test_embed_stores_chunks_for_long_doc():
    pg = _make_pg()
    long_body = " ".join(f"word{i}" for i in range(3500))
    fake_vecs = [[0.1] * 5] * 4  # enough for any batch

    mock_provider = _make_provider(fake_vecs[:1])

    with (
        patch(
            "dewie.enrichment.chunk_embedder.get_embedding_provider", return_value=mock_provider
        ),
        patch("dewie.enrichment.chunk_embedder.chunk_document") as mock_chunk,
    ):
        mock_chunk.return_value = ["chunk text"]
        mock_provider.embed.return_value = [[0.1, 0.2, 0.3]]

        ok = await embed_and_store_chunks(pg, str(uuid4()), "Title", "example.com", long_body)

    assert ok is True
    pg.insert_chunks.assert_called_once()
    pg.mark_chunk_status.assert_called_once()
    args = pg.mark_chunk_status.call_args[0]
    assert args[1] == "chunked"


@pytest.mark.asyncio
async def test_embed_skips_empty_body():
    pg = _make_pg()
    ok = await embed_and_store_chunks(pg, str(uuid4()), "Title", "example.com", "")
    assert ok is True
    pg.mark_chunk_status.assert_called_once()
    assert pg.mark_chunk_status.call_args[0][1] == "skipped"
    pg.insert_chunks.assert_not_called()


@pytest.mark.asyncio
async def test_embed_returns_false_on_provider_error():
    pg = _make_pg()
    body = "word " * 100

    mock_provider = MagicMock()
    mock_provider.embed = AsyncMock(side_effect=RuntimeError("embedding failed"))

    with (
        patch(
            "dewie.enrichment.chunk_embedder.get_embedding_provider", return_value=mock_provider
        ),
        patch("dewie.enrichment.chunk_embedder.chunk_document", return_value=["chunk text"]),
        patch("dewie.enrichment.chunk_embedder._embed_dimensions_for_model", return_value=None),
    ):
        ok = await embed_and_store_chunks(pg, str(uuid4()), "Title", "example.com", body)

    assert ok is False


@pytest.mark.asyncio
async def test_embed_returns_false_on_wrong_vector_count():
    pg = _make_pg()
    body = "word " * 100

    mock_provider = MagicMock()
    mock_provider.embed = AsyncMock(return_value=[])  # empty — wrong count

    with (
        patch(
            "dewie.enrichment.chunk_embedder.get_embedding_provider", return_value=mock_provider
        ),
        patch("dewie.enrichment.chunk_embedder.chunk_document", return_value=["chunk text"]),
    ):
        ok = await embed_and_store_chunks(pg, str(uuid4()), "Title", "example.com", body)

    assert ok is False


# ── run_once ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_once_no_docs():
    pg = _make_pg()
    pg.get_unchunked_docs = AsyncMock(return_value=[])
    result = await run_once(pg)
    assert result == 0


@pytest.mark.asyncio
async def test_run_once_skips_doc_without_body():
    doc_id = str(uuid4())
    pg = _make_pg()
    pg.get_unchunked_docs = AsyncMock(return_value=[(doc_id, "Title", "src", None)])

    with patch("dewie.enrichment.chunk_embedder.load_body", return_value=None):
        result = await run_once(pg)

    assert result == 0
    pg.mark_chunk_status.assert_called_with(UUID(doc_id), "skipped")


@pytest.mark.asyncio
async def test_run_once_processes_doc_with_body():
    doc_id = str(uuid4())
    pg = _make_pg()
    pg.get_unchunked_docs = AsyncMock(return_value=[(doc_id, "Title", "src", None)])
    pg.insert_chunks = AsyncMock()

    fake_provider = MagicMock()
    fake_provider.embed = AsyncMock(return_value=[[0.1, 0.2]])

    with (
        patch("dewie.enrichment.chunk_embedder.load_body", return_value="body text"),
        patch(
            "dewie.enrichment.chunk_embedder.get_embedding_provider", return_value=fake_provider
        ),
        patch("dewie.enrichment.chunk_embedder.chunk_document", return_value=["chunk 1"]),
        patch("dewie.enrichment.chunk_embedder._embed_dimensions_for_model", return_value=None),
    ):
        result = await run_once(pg)

    assert result == 1


@pytest.mark.asyncio
async def test_run_once_skips_ids_in_skip_set():
    doc_id = str(uuid4())
    pg = _make_pg()
    pg.get_unchunked_docs = AsyncMock(return_value=[(doc_id, "Title", "src", "some body")])

    result = await run_once(pg, skip_ids={doc_id})
    assert result == 0
    pg.mark_chunk_status.assert_not_called()


@pytest.mark.asyncio
async def test_run_once_marks_failed_on_embed_error():
    doc_id = str(uuid4())
    pg = _make_pg()
    pg.get_unchunked_docs = AsyncMock(return_value=[(doc_id, "Title", "src", None)])

    with (
        patch("dewie.enrichment.chunk_embedder.load_body", return_value="body text"),
        patch(
            "dewie.enrichment.chunk_embedder.embed_and_store_chunks",
            new=AsyncMock(side_effect=RuntimeError("embed failed")),
        ),
    ):
        result = await run_once(pg)

    assert result == 0
    pg.mark_chunk_status.assert_called_with(UUID(doc_id), "failed")
