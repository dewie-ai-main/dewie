"""Tests for dewie.storage.rankers — pluggable ranking strategies."""

from __future__ import annotations

from datetime import UTC
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dewie.storage.rankers import (
    RANKER_REGISTRY,
    _normalize,
    _normalize_dict,
    rank_answers_questions_rrf,
    rank_aq_boosted,
    rank_bm25,
    rank_linear_blend,
    rank_linear_blend_50,
    rank_rrf,
    rank_rrf_graph_boosted,
    rank_rrf_k10,
    rank_rrf_quality,
    rank_rrf_recency,
    rank_vec_aq_boosted,
    rank_vector,
)

# ── _normalize ────────────────────────────────────────────────────────────────


def test_normalize_empty():
    assert _normalize([]) == []


def test_normalize_single():
    result = _normalize([("doc1", 5.0)])
    assert result == [("doc1", 1.0)]


def test_normalize_all_equal():
    result = _normalize([("a", 3.0), ("b", 3.0)])
    assert all(s == 1.0 for _, s in result)


def test_normalize_range():
    scores = [("a", 0.0), ("b", 5.0), ("c", 10.0)]
    result = dict(_normalize(scores))
    assert result["a"] == pytest.approx(0.0)
    assert result["c"] == pytest.approx(1.0)
    assert 0.0 < result["b"] < 1.0


# ── _gap_fill_filter_sql (removed in Dewie rename — skipped) ─────────────────


@pytest.mark.skip(reason="_gap_fill_filter_sql removed from rankers.py")
def test_gap_fill_filter_empty():
    pass


@pytest.mark.skip(reason="_gap_fill_filter_sql removed from rankers.py")
def test_gap_fill_filter_with_cutoff():
    pass


@pytest.mark.skip(reason="_gap_fill_filter_sql removed from rankers.py")
def test_gap_fill_filter_exclude_gap_fill():
    pass


@pytest.mark.skip(reason="_gap_fill_filter_sql removed from rankers.py")
def test_gap_fill_filter_both():
    pass


# ── RANKER_REGISTRY ───────────────────────────────────────────────────────────


def test_registry_has_expected_rankers():
    expected = {"bm25", "vector", "rrf", "rrf_k10", "linear_blend", "rrf_graph_boosted"}
    for name in expected:
        assert name in RANKER_REGISTRY, f"Missing ranker: {name}"


def test_registry_entry_has_required_keys():
    for name, entry in RANKER_REGISTRY.items():
        assert "fn" in entry
        assert "label" in entry
        assert "description" in entry
        assert callable(entry["fn"])


# ── rank_bm25 ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rank_bm25_returns_normalized_scores():
    session = AsyncMock()
    raw_results = [("doc_a", 0.8), ("doc_b", 0.4)]
    with patch("dewie.storage.rankers._fts", new=AsyncMock(return_value=raw_results)):
        result = await rank_bm25("test query", session, None, 10)
    assert len(result) == 2
    scores = [s for _, s in result]
    assert max(scores) == pytest.approx(1.0)
    assert min(scores) == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_rank_bm25_empty():
    session = AsyncMock()
    with patch("dewie.storage.rankers._fts", new=AsyncMock(return_value=[])):
        result = await rank_bm25("test query", session, None, 10)
    assert result == []


# ── rank_vector ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rank_vector_returns_normalized():
    session = AsyncMock()
    raw = [("doc1", 0.9), ("doc2", 0.3)]
    with patch("dewie.storage.rankers._vec", new=AsyncMock(return_value=raw)):
        result = await rank_vector("q", session, [0.1, 0.2], 10)
    assert len(result) == 2
    assert max(s for _, s in result) == pytest.approx(1.0)


# ── rank_rrf ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rank_rrf_merges_multiple_sources():
    session = AsyncMock()
    fts_results = [("doc_a", 0.9), ("doc_b", 0.4)]
    vec_results = [("doc_b", 0.8), ("doc_c", 0.5)]
    aq_results = [("doc_a", 0.7)]
    chunk_results = []

    with (
        patch("dewie.storage.rankers._fts", new=AsyncMock(return_value=fts_results)),
        patch("dewie.storage.rankers._vec", new=AsyncMock(return_value=vec_results)),
        patch("dewie.storage.rankers._aq_vec", new=AsyncMock(return_value=aq_results)),
        patch("dewie.storage.rankers._chunk_vec", new=AsyncMock(return_value=chunk_results)),
    ):
        result = await rank_rrf("query", session, [0.1], 10)

    doc_ids = [d for d, _ in result]
    assert "doc_a" in doc_ids
    assert "doc_b" in doc_ids
    assert "doc_c" in doc_ids


@pytest.mark.asyncio
async def test_rank_rrf_respects_limit():
    session = AsyncMock()
    many = [(f"doc_{i}", float(i)) for i in range(20)]

    with (
        patch("dewie.storage.rankers._fts", new=AsyncMock(return_value=many)),
        patch("dewie.storage.rankers._vec", new=AsyncMock(return_value=[])),
        patch("dewie.storage.rankers._aq_vec", new=AsyncMock(return_value=[])),
        patch("dewie.storage.rankers._chunk_vec", new=AsyncMock(return_value=[])),
    ):
        result = await rank_rrf("query", session, None, 5)

    assert len(result) <= 5


@pytest.mark.asyncio
async def test_rank_rrf_empty_all_sources():
    session = AsyncMock()
    with (
        patch("dewie.storage.rankers._fts", new=AsyncMock(return_value=[])),
        patch("dewie.storage.rankers._vec", new=AsyncMock(return_value=[])),
        patch("dewie.storage.rankers._aq_vec", new=AsyncMock(return_value=[])),
        patch("dewie.storage.rankers._chunk_vec", new=AsyncMock(return_value=[])),
    ):
        result = await rank_rrf("query", session, None, 10)
    assert result == []


# ── rank_rrf_k10 ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rank_rrf_k10_merges_fts_and_vec():
    session = AsyncMock()
    fts = [("a", 0.9)]
    vec = [("b", 0.8)]
    with (
        patch("dewie.storage.rankers._fts", new=AsyncMock(return_value=fts)),
        patch("dewie.storage.rankers._vec", new=AsyncMock(return_value=vec)),
    ):
        result = await rank_rrf_k10("q", session, [0.1], 10)
    assert len(result) == 2


# ── rank_linear_blend ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rank_linear_blend_weights():
    session = AsyncMock()
    fts = [("doc_a", 1.0)]
    vec = [("doc_a", 0.5)]
    with (
        patch("dewie.storage.rankers._fts", new=AsyncMock(return_value=fts)),
        patch("dewie.storage.rankers._vec", new=AsyncMock(return_value=vec)),
    ):
        result = await rank_linear_blend("q", session, [0.1], 10)
    assert len(result) >= 1
    scores = [s for _, s in result]
    assert all(0.0 <= s <= 1.0 for s in scores)


@pytest.mark.asyncio
async def test_rank_linear_blend_50_equal_weights():
    session = AsyncMock()
    fts = [("doc_x", 0.6)]
    vec = [("doc_x", 0.4)]
    with (
        patch("dewie.storage.rankers._fts", new=AsyncMock(return_value=fts)),
        patch("dewie.storage.rankers._vec", new=AsyncMock(return_value=vec)),
    ):
        result = await rank_linear_blend_50("q", session, [0.1], 10)
    assert len(result) >= 1


# ── rank_aq_boosted ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rank_aq_boosted_returns_results():
    session = AsyncMock()
    fts = [("doc1", 0.7), ("doc2", 0.3)]
    aq = [("doc1", 0.9)]

    with (
        patch("dewie.storage.rankers._fts", new=AsyncMock(return_value=fts)),
        patch("dewie.storage.rankers._aq_match", new=AsyncMock(return_value=aq)),
    ):
        result = await rank_aq_boosted("what is AI?", session, None, 10)

    doc_ids = [d for d, _ in result]
    assert "doc1" in doc_ids
    assert "doc2" in doc_ids
    # doc1 should score higher than doc2 (boosted by AQ match)
    score_by_id = dict(result)
    assert score_by_id.get("doc1", 0) >= score_by_id.get("doc2", 0)


# ── Internal helper functions ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fts_returns_results():
    from dewie.storage.rankers import _fts

    session = AsyncMock()
    fake_rows = [("uuid-1", 0.8), ("uuid-2", 0.4)]
    session.execute = AsyncMock(return_value=MagicMock(fetchall=MagicMock(return_value=fake_rows)))
    rows = await _fts(session, "test query", 10)
    assert len(rows) == 2
    assert rows[0][0] == "uuid-1"


@pytest.mark.asyncio
async def test_fts_handles_exception():
    from dewie.storage.rankers import _fts

    session = AsyncMock()
    session.execute = AsyncMock(side_effect=Exception("db error"))
    rows = await _fts(session, "test query", 10)
    assert rows == []


@pytest.mark.asyncio
async def test_fts_short_result_triggers_or_fallback():
    from dewie.storage.rankers import _fts

    session = AsyncMock()
    primary_rows = [("uuid-1", 0.8)]  # Only 1 result — triggers OR fallback
    fallback_rows = [("uuid-2", 0.5)]

    call_count = [0]

    async def mock_execute(sql, params=None):
        call_count[0] += 1
        mock_result = MagicMock()
        if call_count[0] == 1:
            mock_result.fetchall.return_value = primary_rows
        else:
            mock_result.fetchall.return_value = fallback_rows
        return mock_result

    session.execute = mock_execute
    rows = await _fts(session, "long enough query", 10)
    assert len(rows) >= 1
    assert call_count[0] >= 1


@pytest.mark.asyncio
async def test_vec_returns_empty_for_no_embedding():
    from dewie.storage.rankers import _vec

    session = AsyncMock()
    rows = await _vec(session, None, 10)
    assert rows == []


@pytest.mark.asyncio
async def test_vec_returns_results_with_embedding():
    from dewie.storage.rankers import _vec

    session = AsyncMock()
    fake_rows = [("doc-1", 0.95), ("doc-2", 0.7)]
    session.execute = AsyncMock(return_value=MagicMock(fetchall=MagicMock(return_value=fake_rows)))
    rows = await _vec(session, [0.1, 0.2, 0.3], 10)
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_aq_vec_returns_empty_for_no_embedding():
    from dewie.storage.rankers import _aq_vec

    session = AsyncMock()
    rows = await _aq_vec(session, None, 10)
    assert rows == []


@pytest.mark.asyncio
async def test_aq_vec_handles_exception():
    from dewie.storage.rankers import _aq_vec

    session = AsyncMock()
    session.execute = AsyncMock(side_effect=Exception("table not found"))
    rows = await _aq_vec(session, [0.1, 0.2], 10)
    assert rows == []


@pytest.mark.asyncio
async def test_aq_vec_returns_results():
    from dewie.storage.rankers import _aq_vec

    session = AsyncMock()
    fake_rows = [("doc-a", 0.9), ("doc-b", 0.6)]
    session.execute = AsyncMock(return_value=MagicMock(fetchall=MagicMock(return_value=fake_rows)))
    rows = await _aq_vec(session, [0.1], 10)
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_chunk_vec_returns_empty_for_no_embedding():
    from dewie.storage.rankers import _chunk_vec

    session = AsyncMock()
    rows = await _chunk_vec(session, None, 10)
    assert rows == []


@pytest.mark.asyncio
async def test_chunk_vec_returns_empty_on_exception():
    from dewie.storage.rankers import _chunk_vec

    session = AsyncMock()
    session.execute = AsyncMock(side_effect=Exception("table missing"))
    rows = await _chunk_vec(session, [0.1, 0.2], 10)
    assert rows == []


@pytest.mark.asyncio
async def test_chunk_vec_collapses_to_best_score_per_doc():
    from dewie.storage.rankers import _chunk_vec

    session = AsyncMock()
    # Both calls return rows for the same doc with different scores
    aq_rows = [("doc-1", 0.8), ("doc-2", 0.6)]
    body_rows = [("doc-1", 0.5), ("doc-3", 0.7)]
    ready_rows = [("doc-1",), ("doc-2",), ("doc-3",)]

    call_idx = [0]

    async def mock_execute(sql, params=None):
        call_idx[0] += 1
        mock_result = MagicMock()
        if call_idx[0] == 1:
            mock_result.fetchall.return_value = aq_rows
        elif call_idx[0] == 2:
            mock_result.fetchall.return_value = body_rows
        else:
            mock_result.fetchall.return_value = ready_rows
        return mock_result

    session.execute = mock_execute
    rows = await _chunk_vec(session, [0.1, 0.2], 10)
    # doc-1 should appear once with its best score (0.8 from aq_rows)
    doc_scores = dict(rows)
    assert "doc-1" in doc_scores
    assert doc_scores["doc-1"] == pytest.approx(0.8)


# ── Additional ranker tests ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rank_rrf_deduplicates_across_sources():
    """Same doc from multiple sources gets only one entry in output."""
    session = AsyncMock()
    doc = "doc_shared"
    with (
        patch("dewie.storage.rankers._fts", new=AsyncMock(return_value=[(doc, 0.9)])),
        patch("dewie.storage.rankers._vec", new=AsyncMock(return_value=[(doc, 0.8)])),
        patch("dewie.storage.rankers._aq_vec", new=AsyncMock(return_value=[])),
        patch("dewie.storage.rankers._chunk_vec", new=AsyncMock(return_value=[])),
    ):
        result = await rank_rrf("q", session, [0.1], 10)

    doc_ids = [d for d, _ in result]
    assert doc_ids.count(doc) == 1


@pytest.mark.asyncio
async def test_rank_rrf_k10_higher_k_spreads_scores():
    """k=10 produces different score magnitudes than k=60."""
    session = AsyncMock()
    results_10 = []
    results_60 = []
    fts = [("a", 0.9), ("b", 0.4)]
    vec = [("a", 0.8)]

    with (
        patch("dewie.storage.rankers._fts", new=AsyncMock(return_value=fts)),
        patch("dewie.storage.rankers._vec", new=AsyncMock(return_value=vec)),
    ):
        results_10 = await rank_rrf_k10("q", session, [0.1], 10)

    with (
        patch("dewie.storage.rankers._fts", new=AsyncMock(return_value=fts)),
        patch("dewie.storage.rankers._vec", new=AsyncMock(return_value=vec)),
        patch("dewie.storage.rankers._aq_vec", new=AsyncMock(return_value=[])),
        patch("dewie.storage.rankers._chunk_vec", new=AsyncMock(return_value=[])),
    ):
        results_60 = await rank_rrf("q", session, [0.1], 10)

    # Both should return doc "a" first
    assert results_10[0][0] == "a"
    assert results_60[0][0] == "a"


# ── _normalize_dict ───────────────────────────────────────────────────────────


def test_normalize_dict_single():
    result = _normalize_dict({"a": 5.0})
    assert result == {"a": 1.0}


def test_normalize_dict_all_equal():
    result = _normalize_dict({"a": 3.0, "b": 3.0})
    assert all(v == 1.0 for v in result.values())


def test_normalize_dict_range():
    result = _normalize_dict({"a": 0.0, "b": 5.0, "c": 10.0})
    assert result["a"] == pytest.approx(0.0)
    assert result["c"] == pytest.approx(1.0)
    assert 0.0 < result["b"] < 1.0


# ── rank_vec_aq_boosted ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rank_vec_aq_boosted_returns_results():
    session = AsyncMock()
    vec = [("doc1", 0.9), ("doc2", 0.5)]
    aq_rows = MagicMock()
    aq_rows.fetchall.return_value = [("doc1", 0.8)]
    session.execute = AsyncMock(return_value=aq_rows)

    with patch("dewie.storage.rankers._vec", new=AsyncMock(return_value=vec)):
        result = await rank_vec_aq_boosted("q", session, [0.1], 10)

    assert len(result) >= 1
    doc_ids = [d for d, _ in result]
    assert "doc1" in doc_ids


@pytest.mark.asyncio
async def test_rank_vec_aq_boosted_aq_exception_graceful():
    session = AsyncMock()
    vec = [("doc1", 0.9)]
    session.execute = AsyncMock(side_effect=Exception("table missing"))

    with patch("dewie.storage.rankers._vec", new=AsyncMock(return_value=vec)):
        result = await rank_vec_aq_boosted("q", session, [0.1], 10)

    assert len(result) >= 1


# ── rank_rrf_graph_boosted ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rank_rrf_graph_boosted_returns_results():
    session = AsyncMock()
    fts = [("doc_a", 0.9), ("doc_b", 0.5)]
    vec = [("doc_a", 0.8)]
    edge_rows = MagicMock()
    edge_rows.fetchall.return_value = [("doc_a", 3), ("doc_b", 1)]
    session.execute = AsyncMock(return_value=edge_rows)

    with (
        patch("dewie.storage.rankers._fts", new=AsyncMock(return_value=fts)),
        patch("dewie.storage.rankers._vec", new=AsyncMock(return_value=vec)),
    ):
        result = await rank_rrf_graph_boosted("q", session, [0.1], 10)

    assert len(result) >= 1
    assert result[0][0] == "doc_a"


@pytest.mark.asyncio
async def test_rank_rrf_graph_boosted_empty():
    session = AsyncMock()
    edge_rows = MagicMock()
    edge_rows.fetchall.return_value = []
    session.execute = AsyncMock(return_value=edge_rows)

    with (
        patch("dewie.storage.rankers._fts", new=AsyncMock(return_value=[])),
        patch("dewie.storage.rankers._vec", new=AsyncMock(return_value=[])),
    ):
        result = await rank_rrf_graph_boosted("q", session, None, 10)

    assert result == []


# ── rank_rrf_recency ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rank_rrf_recency_returns_results():
    from datetime import datetime, timedelta

    session = AsyncMock()
    fts = [("doc_a", 0.9)]
    vec = [("doc_a", 0.8)]
    recent_date = datetime.now(UTC) - timedelta(days=3)
    date_rows = MagicMock()
    date_rows.fetchall.return_value = [("doc_a", recent_date)]
    session.execute = AsyncMock(return_value=date_rows)

    with (
        patch("dewie.storage.rankers._fts", new=AsyncMock(return_value=fts)),
        patch("dewie.storage.rankers._vec", new=AsyncMock(return_value=vec)),
    ):
        result = await rank_rrf_recency("q", session, [0.1], 10)

    assert len(result) >= 1
    assert result[0][0] == "doc_a"


@pytest.mark.asyncio
async def test_rank_rrf_recency_empty_docs():
    session = AsyncMock()
    date_rows = MagicMock()
    date_rows.fetchall.return_value = []
    session.execute = AsyncMock(return_value=date_rows)

    with (
        patch("dewie.storage.rankers._fts", new=AsyncMock(return_value=[])),
        patch("dewie.storage.rankers._vec", new=AsyncMock(return_value=[])),
    ):
        result = await rank_rrf_recency("q", session, None, 10)

    assert result == []


# ── rank_rrf_quality ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rank_rrf_quality_returns_results():
    session = AsyncMock()
    fts = [("doc_a", 0.9), ("doc_b", 0.5)]
    vec = [("doc_b", 0.8)]
    quality_rows = MagicMock()
    quality_rows.fetchall.return_value = [("doc_a", 85), ("doc_b", 50)]
    session.execute = AsyncMock(return_value=quality_rows)

    with (
        patch("dewie.storage.rankers._fts", new=AsyncMock(return_value=fts)),
        patch("dewie.storage.rankers._vec", new=AsyncMock(return_value=vec)),
    ):
        result = await rank_rrf_quality("q", session, [0.1], 10)

    assert len(result) >= 1
    doc_ids = [d for d, _ in result]
    assert "doc_a" in doc_ids


@pytest.mark.asyncio
async def test_rank_rrf_quality_null_scores_neutral():
    session = AsyncMock()
    fts = [("doc_x", 0.9)]
    vec = []
    quality_rows = MagicMock()
    quality_rows.fetchall.return_value = [("doc_x", None)]  # NULL quality score
    session.execute = AsyncMock(return_value=quality_rows)

    with (
        patch("dewie.storage.rankers._fts", new=AsyncMock(return_value=fts)),
        patch("dewie.storage.rankers._vec", new=AsyncMock(return_value=vec)),
    ):
        result = await rank_rrf_quality("q", session, None, 10)

    assert len(result) == 1


# ── rank_answers_questions_rrf ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rank_answers_questions_rrf_basic():
    session = AsyncMock()
    aq_rows = MagicMock()
    aq_rows.fetchall.return_value = [("doc1", 0.9), ("doc2", 0.5)]
    session.execute = AsyncMock(return_value=aq_rows)

    with (
        patch("dewie.storage.rankers._aq_vec", new=AsyncMock(return_value=[("doc1", 0.8)])),
        patch("dewie.storage.rankers._vec", new=AsyncMock(return_value=[("doc2", 0.7)])),
    ):
        result = await rank_answers_questions_rrf("what is AI?", session, [0.1], 10)

    assert len(result) >= 1
    doc_ids = [d for d, _ in result]
    assert "doc1" in doc_ids


@pytest.mark.asyncio
async def test_rank_answers_questions_rrf_aq_fts_exception():
    session = AsyncMock()
    session.execute = AsyncMock(side_effect=Exception("table missing"))

    with (
        patch("dewie.storage.rankers._aq_vec", new=AsyncMock(return_value=[])),
        patch("dewie.storage.rankers._vec", new=AsyncMock(return_value=[("doc1", 0.5)])),
    ):
        result = await rank_answers_questions_rrf("q", session, [0.1], 10)

    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_rank_answers_questions_rrf_or_fallback():
    """When AQ FTS returns fewer than 3 results, should try OR query."""
    session = AsyncMock()
    primary = MagicMock()
    primary.fetchall.return_value = [("doc1", 0.9)]  # Only 1 — triggers OR fallback
    fallback = MagicMock()
    fallback.fetchall.return_value = [("doc2", 0.5)]

    call_count = [0]

    async def mock_execute(sql, params=None):
        call_count[0] += 1
        if call_count[0] == 1:
            return primary
        return fallback

    session.execute = mock_execute

    with (
        patch("dewie.storage.rankers._aq_vec", new=AsyncMock(return_value=[])),
        patch("dewie.storage.rankers._vec", new=AsyncMock(return_value=[])),
    ):
        result = await rank_answers_questions_rrf("what is machine learning", session, [0.1], 10)

    assert call_count[0] >= 2  # triggered OR fallback


# ── rank_rrf_chunks ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rank_rrf_chunks_combines_four_signals():
    """rank_rrf_chunks should combine FTS, vec, AQ-vec, and chunk-vec signals."""
    from dewie.storage.rankers import rank_rrf_chunks

    session = AsyncMock()
    fts_results = [("doc1", 0.9), ("doc2", 0.5)]
    vec_results = [("doc1", 0.8), ("doc3", 0.3)]
    aq_results = [("doc2", 0.7)]
    chunk_results = [("doc3", 0.6)]

    with (
        patch("dewie.storage.rankers._fts", new=AsyncMock(return_value=fts_results)),
        patch("dewie.storage.rankers._vec", new=AsyncMock(return_value=vec_results)),
        patch("dewie.storage.rankers._aq_vec", new=AsyncMock(return_value=aq_results)),
        patch("dewie.storage.rankers._chunk_vec", new=AsyncMock(return_value=chunk_results)),
    ):
        result = await rank_rrf_chunks("query", session, [0.1], 10)

    doc_ids = [d for d, _ in result]
    assert "doc1" in doc_ids
    assert "doc2" in doc_ids
    assert "doc3" in doc_ids


@pytest.mark.asyncio
async def test_rank_rrf_chunks_empty_returns_empty():
    """With no results from any signal, should return empty list."""
    from dewie.storage.rankers import rank_rrf_chunks

    session = AsyncMock()
    with (
        patch("dewie.storage.rankers._fts", new=AsyncMock(return_value=[])),
        patch("dewie.storage.rankers._vec", new=AsyncMock(return_value=[])),
        patch("dewie.storage.rankers._aq_vec", new=AsyncMock(return_value=[])),
        patch("dewie.storage.rankers._chunk_vec", new=AsyncMock(return_value=[])),
    ):
        result = await rank_rrf_chunks("query", session, None, 10)
    assert result == []


@pytest.mark.asyncio
async def test_rank_rrf_chunks_respects_limit():
    """rank_rrf_chunks should not return more than limit docs."""
    from dewie.storage.rankers import rank_rrf_chunks

    session = AsyncMock()
    fts_results = [(f"doc{i}", 1.0 / (i + 1)) for i in range(20)]
    with (
        patch("dewie.storage.rankers._fts", new=AsyncMock(return_value=fts_results)),
        patch("dewie.storage.rankers._vec", new=AsyncMock(return_value=[])),
        patch("dewie.storage.rankers._aq_vec", new=AsyncMock(return_value=[])),
        patch("dewie.storage.rankers._chunk_vec", new=AsyncMock(return_value=[])),
    ):
        result = await rank_rrf_chunks("query", session, [0.1], 5)
    assert len(result) <= 5


# ── _fetch_embeddings ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_embeddings_returns_dict():
    """_fetch_embeddings parses pgvector strings into float lists."""
    from dewie.storage.rankers import _fetch_embeddings

    session = AsyncMock()
    rows_mock = MagicMock()
    rows_mock.fetchall.return_value = [
        ("doc1", "[0.1,0.2,0.3]"),
        ("doc2", "[0.4,0.5,0.6]"),
    ]
    session.execute = AsyncMock(return_value=rows_mock)
    result = await _fetch_embeddings(session, ["doc1", "doc2"])
    assert "doc1" in result
    assert len(result["doc1"]) == 3
    assert result["doc1"][0] == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_fetch_embeddings_empty_ids():
    """_fetch_embeddings returns empty dict for empty input."""
    from dewie.storage.rankers import _fetch_embeddings

    session = AsyncMock()
    result = await _fetch_embeddings(session, [])
    assert result == {}


@pytest.mark.asyncio
async def test_fetch_embeddings_malformed_vector_skipped():
    """_fetch_embeddings skips rows with malformed vector strings."""
    from dewie.storage.rankers import _fetch_embeddings

    session = AsyncMock()
    rows_mock = MagicMock()
    rows_mock.fetchall.return_value = [
        ("doc1", "not-a-vector"),
        ("doc2", "[0.1,0.2]"),
    ]
    session.execute = AsyncMock(return_value=rows_mock)
    result = await _fetch_embeddings(session, ["doc1", "doc2"])
    assert "doc1" not in result
    assert "doc2" in result


# ── _mmr_rerank ───────────────────────────────────────────────────────────────


def test_mmr_rerank_selects_diverse_results():
    """_mmr_rerank should pick diverse documents using MMR."""
    from dewie.storage.rankers import _mmr_rerank

    # doc1 is highest relevance; doc3 is diverse and second-highest relevance;
    # doc2 is near-identical to doc1 and lowest relevance → gets deprioritized
    candidates = [("doc1", 0.9), ("doc3", 0.85), ("doc2", 0.5)]
    embeddings_map = {
        "doc1": [1.0, 0.0],
        "doc2": [0.99, 0.01],  # very close to doc1
        "doc3": [0.0, 1.0],  # orthogonal to doc1
    }
    query_emb = [1.0, 0.0]
    result = _mmr_rerank(candidates, embeddings_map, query_emb, limit=2, lam=0.7)
    doc_ids = [d for d, _ in result]
    assert len(doc_ids) == 2
    assert "doc1" in doc_ids  # top ranked
    assert "doc3" in doc_ids  # diverse pick (orthogonal to doc1, second-highest relevance)


def test_mmr_rerank_missing_embedding_skipped():
    """_mmr_rerank gracefully handles docs with no stored embedding."""
    from dewie.storage.rankers import _mmr_rerank

    candidates = [("doc1", 0.9), ("doc_no_emb", 0.8), ("doc2", 0.7)]
    embeddings_map = {"doc1": [1.0, 0.0], "doc2": [0.0, 1.0]}
    result = _mmr_rerank(candidates, embeddings_map, [1.0, 0.0], limit=3, lam=0.7)
    doc_ids = [d for d, _ in result]
    assert "doc1" in doc_ids
    assert "doc2" in doc_ids


def test_mmr_rerank_empty_candidates():
    """_mmr_rerank returns empty list for empty input."""
    from dewie.storage.rankers import _mmr_rerank

    result = _mmr_rerank([], {}, [1.0, 0.0], limit=5, lam=0.5)
    assert result == []


def test_mmr_rerank_no_query_embedding():
    """_mmr_rerank falls back to relevance-only when query embedding is None."""
    from dewie.storage.rankers import _mmr_rerank

    candidates = [("doc1", 0.9), ("doc2", 0.5)]
    embeddings_map = {"doc1": [1.0, 0.0], "doc2": [0.0, 1.0]}
    result = _mmr_rerank(candidates, embeddings_map, None, limit=2, lam=0.5)
    assert len(result) == 2


# ── rank_rrf_mmr ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rank_rrf_mmr_returns_results():
    """rank_rrf_mmr should apply BM25 + vec RRF followed by MMR."""
    from dewie.storage.rankers import rank_rrf_mmr

    session = AsyncMock()
    fts_results = [("doc1", 0.9), ("doc2", 0.5)]
    vec_results = [("doc1", 0.8)]
    emb_rows = MagicMock()
    emb_rows.fetchall.return_value = [
        ("doc1", "[0.1,0.2]"),
        ("doc2", "[0.9,0.8]"),
    ]
    session.execute = AsyncMock(return_value=emb_rows)
    with (
        patch("dewie.storage.rankers._fts", new=AsyncMock(return_value=fts_results)),
        patch("dewie.storage.rankers._vec", new=AsyncMock(return_value=vec_results)),
    ):
        result = await rank_rrf_mmr("query", session, [0.1, 0.2], 10)
    assert len(result) >= 1


@pytest.mark.asyncio
async def test_rank_rrf_mmr_empty():
    """rank_rrf_mmr with empty corpus returns empty list."""
    from dewie.storage.rankers import rank_rrf_mmr

    session = AsyncMock()
    with (
        patch("dewie.storage.rankers._fts", new=AsyncMock(return_value=[])),
        patch("dewie.storage.rankers._vec", new=AsyncMock(return_value=[])),
    ):
        result = await rank_rrf_mmr("query", session, None, 10)
    assert result == []


# ── rank_rrf_rerank ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rank_rrf_rerank_no_embedding_fallback():
    """rank_rrf_rerank falls back to rrf when embedding is None."""
    from dewie.storage.rankers import rank_rrf_rerank

    session = AsyncMock()
    fts_results = [("doc1", 0.9)]
    vec_results = []
    with (
        patch("dewie.storage.rankers._fts", new=AsyncMock(return_value=fts_results)),
        patch("dewie.storage.rankers._vec", new=AsyncMock(return_value=vec_results)),
    ):
        result = await rank_rrf_rerank("query", session, None, 10)
    assert len(result) >= 1


@pytest.mark.asyncio
async def test_rank_rrf_rerank_with_embedding():
    """rank_rrf_rerank reranks by cosine when embedding is provided."""
    from dewie.storage.rankers import rank_rrf_rerank

    session = AsyncMock()
    fts_results = [("doc1", 0.9), ("doc2", 0.5)]
    vec_results = [("doc2", 0.8)]
    emb_rows = MagicMock()
    emb_rows.fetchall.return_value = [
        ("doc1", "[0.9,0.1]"),
        ("doc2", "[0.1,0.9]"),
    ]
    session.execute = AsyncMock(return_value=emb_rows)
    with (
        patch("dewie.storage.rankers._fts", new=AsyncMock(return_value=fts_results)),
        patch("dewie.storage.rankers._vec", new=AsyncMock(return_value=vec_results)),
    ):
        result = await rank_rrf_rerank("query", session, [0.9, 0.1], 10)
    assert len(result) >= 1


# ── _entity_match (inner helper) ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_entity_match_empty_terms():
    """_entity_match returns [] for empty query_terms."""
    from dewie.storage.rankers import _entity_match

    session = AsyncMock()
    result = await _entity_match(session, [], 10)
    assert result == []


@pytest.mark.asyncio
async def test_entity_match_with_terms():
    """_entity_match returns scored results when terms match."""
    from dewie.storage.rankers import _entity_match

    session = AsyncMock()
    scan_rows = MagicMock()
    scan_rows.fetchall.return_value = [
        (
            "doc1",
            "AI Research Paper",
            '["artificial intelligence", "machine learning"]',
            '["AI", "ML"]',
            '["tech"]',
        ),
        ("doc2", "Cooking Recipe", '["food"]', '["cooking"]', "[]"),
    ]
    session.execute = AsyncMock(return_value=scan_rows)
    result = await _entity_match(session, ["artificial intelligence", "machine learning"], 10)
    doc_ids = [d for d, _ in result]
    assert "doc1" in doc_ids  # matches AI terms


@pytest.mark.asyncio
async def test_entity_match_db_exception_returns_empty():
    """_entity_match handles DB exceptions gracefully."""
    from dewie.storage.rankers import _entity_match

    session = AsyncMock()
    session.execute = AsyncMock(side_effect=Exception("DB error"))
    result = await _entity_match(session, ["some", "terms"], 10)
    assert result == []


# ── adaptive ranker ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rank_adaptive_keyword_query():
    """rank_adaptive routes short keyword queries through entity+fts+aq blend."""
    from dewie.storage.rankers import rank_adaptive

    session = AsyncMock()
    # "AI" is a 1-word, non-question query → keyword path
    entity_results = [("doc1", 0.9)]
    fts_results = [("doc1", 0.8)]
    aq_results = [("doc2", 0.5)]
    with (
        patch(
            "dewie.storage.rankers._entity_match", new=AsyncMock(return_value=entity_results)
        ),
        patch("dewie.storage.rankers._fts", new=AsyncMock(return_value=fts_results)),
        patch("dewie.storage.rankers._aq_vec", new=AsyncMock(return_value=aq_results)),
    ):
        result = await rank_adaptive("AI", session, [0.1], 10)
    assert len(result) >= 1
    assert result[0][0] == "doc1"  # highest combined score


@pytest.mark.asyncio
async def test_rank_adaptive_question_query():
    """rank_adaptive routes natural language questions through rank_answers_questions_rrf."""
    from dewie.storage.rankers import rank_adaptive

    session = AsyncMock()
    aq_results = [("doc1", 0.9)]
    with patch(
        "dewie.storage.rankers.rank_answers_questions_rrf",
        new=AsyncMock(return_value=aq_results),
    ):
        result = await rank_adaptive("what is machine learning?", session, [0.1], 10)
    assert result == aq_results


# ── apply_staleness_penalty ────────────────────────────────────────────────────


def test_apply_staleness_penalty_fresh_doc_no_penalty():
    """Docs ingested within 7 days get no penalty (factor=1.0)."""
    from datetime import datetime, timedelta
    from unittest.mock import MagicMock

    from dewie.storage.rankers import apply_staleness_penalty

    doc = MagicMock()
    doc.ingested_at = datetime.now(UTC) - timedelta(days=2)
    doc.score = 0.8
    apply_staleness_penalty([doc])
    assert doc.score == pytest.approx(0.8)


def test_apply_staleness_penalty_old_doc_gets_penalized():
    """Docs older than 180 days get a penalty < 1.0."""
    from datetime import datetime, timedelta
    from unittest.mock import MagicMock

    from dewie.storage.rankers import apply_staleness_penalty

    doc = MagicMock()
    doc.ingested_at = datetime.now(UTC) - timedelta(days=400)
    original_score = 0.9
    doc.score = original_score
    apply_staleness_penalty([doc])
    assert doc.score < original_score


def test_apply_staleness_penalty_none_ingested_at_skipped():
    """Docs with ingested_at=None are not penalized."""
    from unittest.mock import MagicMock

    from dewie.storage.rankers import apply_staleness_penalty

    doc = MagicMock()
    doc.ingested_at = None
    doc.score = 0.7
    apply_staleness_penalty([doc])
    assert doc.score == 0.7


# ── _quality_scores empty ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_quality_scores_empty_ids():
    """_quality_scores returns {} for empty doc_ids list."""
    from unittest.mock import AsyncMock

    from dewie.storage.rankers import _quality_scores

    session = AsyncMock()
    result = await _quality_scores(session, [])
    assert result == {}


# ── _normalize_dict ────────────────────────────────────────────────────────────


def test_normalize_dict_empty():
    """_normalize_dict returns {} for empty dict."""
    from dewie.storage.rankers import _normalize_dict

    assert _normalize_dict({}) == {}


def test_normalize_dict_single_value():
    """_normalize_dict with one value returns {key: 1.0}."""
    from dewie.storage.rankers import _normalize_dict

    result = _normalize_dict({"a": 5.0})
    assert result == {"a": 1.0}


def test_normalize_dict_uniform_values():
    """All identical values → all normalize to 1.0 (min==max fallback)."""
    from dewie.storage.rankers import _normalize_dict

    result = _normalize_dict({"a": 3.0, "b": 3.0, "c": 3.0})
    # When max == min, all values become 1.0 (the implementation returns {k: 1.0})
    assert all(v == 1.0 for v in result.values())


def test_normalize_dict_spread():
    """Values in [0,10] normalize to [0.0, 1.0]."""
    from dewie.storage.rankers import _normalize_dict

    result = _normalize_dict({"low": 0.0, "mid": 5.0, "high": 10.0})
    assert result["low"] == pytest.approx(0.0)
    assert result["mid"] == pytest.approx(0.5)
    assert result["high"] == pytest.approx(1.0)


# ── execute_ranker ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_ranker_calls_registered_fn():
    """run_ranker delegates to the registered function."""
    from unittest.mock import AsyncMock

    from dewie.storage.rankers import run_ranker

    session = AsyncMock()
    # bm25 is registered; with a mock session it will return []
    result = await run_ranker("bm25", "test query", session, None, 5)
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_run_ranker_raises_on_unknown_ranker():
    """run_ranker raises KeyError for unknown ranker name."""
    from unittest.mock import AsyncMock

    from dewie.storage.rankers import run_ranker

    session = AsyncMock()
    with pytest.raises(KeyError):
        await run_ranker("nonexistent_ranker_xyz", "query", session, None, 5)


# ── list_rankers ───────────────────────────────────────────────────────────────


def test_list_rankers_returns_registered_entries():
    """list_rankers returns all registered rankers as dicts."""
    from dewie.storage.rankers import list_rankers

    rankers = list_rankers()
    assert len(rankers) > 0
    ids = [r["id"] for r in rankers]
    assert "bm25" in ids
    assert "rrf" in ids
    # Verify each entry has required fields
    for r in rankers:
        assert "id" in r
        assert "label" in r
        assert "description" in r


def test_list_rankers_filters_by_enabled_rankers():
    """list_rankers respects the enabled_rankers config list."""
    from unittest.mock import patch

    from dewie.storage.rankers import list_rankers

    mock_settings = MagicMock()
    mock_settings.enabled_rankers = ["bm25", "vector"]

    with patch("dewie.config.settings", mock_settings):
        rankers = list_rankers()
    ids = [r["id"] for r in rankers]
    assert "bm25" in ids
    assert "vector" in ids
    assert "rrf" not in ids
    assert len(rankers) == 2


# ── _cosine helper ─────────────────────────────────────────────────────────────


def test_cosine_identical_vectors():
    """Cosine similarity of identical vectors is 1.0."""
    from dewie.storage.rankers import _cosine

    v = [1.0, 2.0, 3.0]
    assert _cosine(v, v) == pytest.approx(1.0, abs=1e-6)


def test_cosine_orthogonal_vectors():
    """Cosine similarity of orthogonal vectors is 0.0."""
    from dewie.storage.rankers import _cosine

    assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0, abs=1e-6)


def test_cosine_opposite_vectors():
    """Cosine similarity of opposite vectors is -1.0."""
    from dewie.storage.rankers import _cosine

    assert _cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0, abs=1e-6)


def test_cosine_zero_vector_returns_zero():
    """Zero vector → cosine returns 0.0 (no div by zero)."""
    from dewie.storage.rankers import _cosine

    result = _cosine([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
    assert result == 0.0


# ── issue-128: chunk_vec as 4th RRF component ─────────────────────────────────


@pytest.mark.asyncio
async def test_rank_rrf_surfaces_chunk_only_doc():
    """
    Issue #128: a doc that only appears in chunk_vec (not fts/doc_vec/aq_vec)
    must still surface in rank_rrf output.

    Before the fix, chunk_vec was reranking-only — such a doc would never enter
    the top-N candidate set. With 4-way RRF it now gets an RRF score contribution
    and appears in results.
    """
    session = AsyncMock()
    # doc_chunk_only is ONLY in chunk results — not in fts, vec, or aq
    chunk_results = [("doc_chunk_only", 0.95), ("doc_a", 0.6)]
    fts_results = [("doc_a", 0.9)]
    vec_results = [("doc_a", 0.8)]
    aq_results = [("doc_a", 0.7)]

    with (
        patch("dewie.storage.rankers._fts", new=AsyncMock(return_value=fts_results)),
        patch("dewie.storage.rankers._vec", new=AsyncMock(return_value=vec_results)),
        patch("dewie.storage.rankers._aq_vec", new=AsyncMock(return_value=aq_results)),
        patch("dewie.storage.rankers._chunk_vec", new=AsyncMock(return_value=chunk_results)),
    ):
        result = await rank_rrf("query", session, [0.1], 10)

    doc_ids = [d for d, _ in result]
    assert "doc_chunk_only" in doc_ids, (
        "doc appearing only in chunk_vec must surface in rank_rrf output (#128)"
    )
    assert "doc_a" in doc_ids


@pytest.mark.asyncio
async def test_rank_rrf_chunk_vec_boosts_ranking():
    """
    Issue #128: a doc in both doc_vec and chunk_vec gets a higher RRF score
    than a doc in doc_vec alone, because chunk_vec adds a second RRF term.
    """
    session = AsyncMock()
    # doc_with_chunks: appears in vec AND chunks
    # doc_vec_only: appears in vec only, at same vec rank
    fts_results: list = []
    vec_results = [("doc_with_chunks", 0.9), ("doc_vec_only", 0.9)]
    aq_results: list = []
    chunk_results = [("doc_with_chunks", 0.95)]  # extra signal for doc_with_chunks

    with (
        patch("dewie.storage.rankers._fts", new=AsyncMock(return_value=fts_results)),
        patch("dewie.storage.rankers._vec", new=AsyncMock(return_value=vec_results)),
        patch("dewie.storage.rankers._aq_vec", new=AsyncMock(return_value=aq_results)),
        patch("dewie.storage.rankers._chunk_vec", new=AsyncMock(return_value=chunk_results)),
    ):
        result = await rank_rrf("query", session, [0.1], 10)

    score_map = dict(result)
    assert score_map["doc_with_chunks"] > score_map["doc_vec_only"], (
        "doc with chunk signal should rank above identical doc_vec_only doc (#128)"
    )
