"""Tests for query.py models and constants."""

from __future__ import annotations

import pytest

# ── SearchRequest model ────────────────────────────────────────────────────────


def test_search_request_defaults():
    from dewie.config import settings
    from dewie.models.query import SearchRequest

    req = SearchRequest(query="machine learning")
    assert req.query == "machine learning"
    assert req.limit == 10
    assert req.ranker == settings.query_default_ranker


def test_search_request_custom():
    from dewie.models.query import SearchRequest

    req = SearchRequest(query="python", limit=5, ranker="bm25")
    assert req.limit == 5
    assert req.ranker == "bm25"


# ── SearchResult model ─────────────────────────────────────────────────────────


def test_search_result_minimal():
    from dewie.models.query import SearchResult

    r = SearchResult(
        doc_id="d1",
        doc_type="blog_post",
        url="https://example.com",
        title="Test",
        summary="Summary",
        source="web",
        ingested_at="2024-01-01T00:00:00Z",
        topics=[],
        keywords=[],
        entities=[],
        sentiment=0.0,
        score=0.5,
        answers_questions=[],
        edge_count=0,
        enrichment_quality_score=None,
        reading_level=None,
        chunk_match=None,
        chunk_score=None,
    )
    assert r.doc_id == "d1"
    assert r.score == 0.5


# ── ResultConfidence model ─────────────────────────────────────────────────────


def test_result_confidence_model():
    from dewie.models.query import ResultConfidence

    c = ResultConfidence(
        confidence_level="high",
        score_gap=0.1,
        aq_coverage_ratio=0.8,
        edge_density=0.5,
        complexity="lookup",
        suggested_action="none",
    )
    assert c.confidence_level == "high"
    assert c.score_gap == pytest.approx(0.1)


# ── AgentQueryRequest / AgentToolCall / AgentQueryResponse ────────────────────


def test_agent_query_request_defaults():
    from dewie.api.routes.query import AgentQueryRequest

    req = AgentQueryRequest(query="What is AI?")
    assert req.model == ""
    assert req.max_hops == 5


def test_agent_tool_call_model():
    from dewie.api.routes.query import AgentToolCall

    tc = AgentToolCall(tool="dewie_search", args={"query": "AI"}, response_preview="Result 1")
    assert tc.tool == "dewie_search"
    assert tc.was_read is None


def test_agent_query_response_model():
    from dewie.api.routes.query import AgentQueryResponse

    resp = AgentQueryResponse(
        query="What is AI?",
        answer="AI is...",
        searches=1,
        reads=1,
        tool_calls=[],
        docs_read=[],
    )
    assert resp.query_id is None
    assert resp.searches == 1


# ── BlindQueryRequest / BlindQueryResponse ─────────────────────────────────────


def test_blind_query_request_defaults():
    from dewie.api.routes.query import BlindQueryRequest

    req = BlindQueryRequest(query="What is AI?")
    assert req.model == ""


def test_blind_query_response_model():
    from dewie.api.routes.query import BlindQueryResponse

    resp = BlindQueryResponse(query="What is AI?", answer="AI is...", model="gpt-4o")
    assert resp.answer == "AI is..."


# ── _AGENT_SYSTEM constant ─────────────────────────────────────────────────────


def test_agent_system_prompt_contains_rules():
    from dewie.api.routes.query import _AGENT_SYSTEM

    assert "dewie_search" in _AGENT_SYSTEM
    assert "dewie_read" in _AGENT_SYSTEM


# ── _AGENT_TOOLS structure ─────────────────────────────────────────────────────


def test_agent_tools_structure():
    from dewie.api.routes.query import _AGENT_TOOLS

    assert len(_AGENT_TOOLS) >= 2
    names = [t["function"]["name"] for t in _AGENT_TOOLS]
    assert "dewie_search" in names
    assert "dewie_read" in names


# ── CategoryHint model ────────────────────────────────────────────────────────


def test_category_hint_model():
    from dewie.models.query import CategoryHint

    hint = CategoryHint(result_count=5, corpus_count=100, suggested=False)
    assert hint.result_count == 5
    assert hint.suggested is False


# ── BenchmarkResult / BenchmarkResponse ──────────────────────────────────────


def test_benchmark_result_model():
    from dewie.models.query import BenchmarkResult

    result = BenchmarkResult(
        doc_id="d1",
        title="Test",
        doc_score=0.8,
        standard_rank=1,
        reranked_rank=1,
        rank_change=0,
    )
    assert result.doc_score == pytest.approx(0.8)
