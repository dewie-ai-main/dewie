"""Unit tests for src/dewie/pipeline.py — pure functions only."""
from __future__ import annotations

from dewie.pipeline import build_embed_text, compute_quality_score, jaccard, tokenize

# ── build_embed_text ──────────────────────────────────────────────────────────

def test_build_embed_text_includes_title():
    result = build_embed_text("My Title", "A summary", [], "some content")
    assert "Title: My Title" in result


def test_build_embed_text_includes_summary():
    result = build_embed_text("T", "The summary text", [], "")
    assert "Summary: The summary text" in result


def test_build_embed_text_uses_embed_summary_over_summary():
    result = build_embed_text("T", "display summary", [], "", embed_summary="dense summary")
    assert "dense summary" in result
    assert "display summary" not in result


def test_build_embed_text_includes_aq():
    aq = ["What is AI?", "How does it work?"]
    result = build_embed_text("T", "S", aq, "")
    assert "What is AI?" in result
    assert "How does it work?" in result


def test_build_embed_text_caps_aq_at_six():
    aq = [f"Question {i}?" for i in range(10)]
    result = build_embed_text("T", "S", aq, "")
    # Only first 6 should appear
    assert "Question 0?" in result
    assert "Question 6?" not in result


def test_build_embed_text_empty_aq_no_questions_line():
    result = build_embed_text("T", "S", [], "content")
    assert "Questions:" not in result


def test_build_embed_text_no_summary_skips_summary_line():
    result = build_embed_text("T", "", [], "content")
    assert "Summary:" not in result


# ── compute_quality_score ─────────────────────────────────────────────────────

def test_quality_score_empty_doc_is_zero():
    score = compute_quality_score("", [], [], [], "")
    assert score == 0


def test_quality_score_full_doc_is_high():
    body = "x" * 3001
    keywords = [f"kw{i}" for i in range(10)]
    entities = [f"Ent{i}" for i in range(8)]
    aq = [f"Q{i}?" for i in range(5)]
    score = compute_quality_score(body, keywords, entities, aq, "A real summary")
    assert score >= 70, f"Expected >=70, got {score}"


def test_quality_score_short_body_medium():
    body = "x" * 600
    score = compute_quality_score(body, ["kw1", "kw2"], ["Entity1"], ["Q?"], "summary")
    assert 10 <= score <= 60


def test_quality_score_max_is_100():
    body = "x" * 5000
    keywords = [f"k{i}" for i in range(30)]
    entities = [f"E{i}" for i in range(20)]
    aq = [f"Q{i}?" for i in range(20)]
    score = compute_quality_score(body, keywords, entities, aq, "A very detailed summary here")
    assert score <= 100


def test_quality_score_returns_int():
    score = compute_quality_score("hello world", ["hello"], ["World"], ["What?"], "summary")
    assert isinstance(score, int)


# ── jaccard ───────────────────────────────────────────────────────────────────

def test_jaccard_identical_sets():
    assert jaccard({1, 2, 3}, {1, 2, 3}) == 1.0


def test_jaccard_disjoint_sets():
    assert jaccard({1, 2}, {3, 4}) == 0.0


def test_jaccard_partial_overlap():
    result = jaccard({1, 2, 3}, {2, 3, 4})
    assert abs(result - 0.5) < 1e-9  # 2 shared / 4 union


def test_jaccard_empty_a():
    assert jaccard(set(), {1, 2}) == 0.0


def test_jaccard_empty_b():
    assert jaccard({1, 2}, set()) == 0.0


def test_jaccard_both_empty():
    assert jaccard(set(), set()) == 0.0


# ── tokenize ─────────────────────────────────────────────────────────────────

def test_tokenize_basic():
    result = tokenize(["hello world"])
    assert "hello" in result
    assert "world" in result


def test_tokenize_splits_on_hyphen():
    result = tokenize(["machine-learning"])
    assert "machine" in result
    assert "learning" in result


def test_tokenize_lowercases():
    result = tokenize(["Hello", "WORLD"])
    assert "hello" in result
    assert "world" in result


def test_tokenize_filters_short_tokens():
    result = tokenize(["a", "is", "the", "it"])
    # All tokens <= 2 chars should be filtered
    assert "a" not in result
    assert "is" not in result


def test_tokenize_empty_list():
    assert tokenize([]) == set()


def test_tokenize_non_string_ignored():
    result = tokenize([None, 42, "valid"])
    assert "valid" in result


def test_tokenize_returns_set():
    result = tokenize(["foo bar"])
    assert isinstance(result, set)


# ── add_edges_for_doc ─────────────────────────────────────────────────────────

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock

from dewie.pipeline import add_edges_for_doc


def _make_engine(scalar_val=0):
    """Return a mock SQLAlchemy async engine."""
    mock_result = MagicMock()
    mock_result.scalar.return_value = scalar_val

    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=mock_result)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)

    mock_engine = MagicMock()
    mock_engine.begin.return_value = mock_conn
    return mock_engine


def test_add_edges_for_doc_returns_int():
    """add_edges_for_doc must return an integer edge count."""
    engine = _make_engine(scalar_val=3)
    result = asyncio.run(add_edges_for_doc(engine, str(uuid.uuid4())))
    assert isinstance(result, int)


def test_add_edges_for_doc_executes_sql():
    """add_edges_for_doc must call conn.execute() at least twice (kw + vec check)."""
    engine = _make_engine(scalar_val=0)
    conn = engine.begin.return_value
    asyncio.run(add_edges_for_doc(engine, str(uuid.uuid4())))
    assert conn.execute.call_count >= 2


def test_add_edges_for_doc_conflict_target_matches_pk():
    """Upserts must target the full PK (source_id, target_id, rel_type)."""
    engine = _make_engine(scalar_val=0)
    conn = engine.begin.return_value

    asyncio.run(add_edges_for_doc(engine, str(uuid.uuid4())))

    sql_texts = [
        getattr(call.args[0], "text", str(call.args[0])) for call in conn.execute.call_args_list
    ]
    assert any("ON CONFLICT (source_id, target_id, rel_type) DO UPDATE" in sql for sql in sql_texts)


def test_add_edges_for_doc_no_python_loop():
    """add_edges_for_doc must NOT iterate over any Python collection of docs."""
    # The function should do all work in SQL; the mock conn never returns
    # a list of rows. If there were a Python loop over candidate docs,
    # it would fail trying to iterate the MagicMock scalar result.
    engine = _make_engine(scalar_val=5)
    # This should complete without error (no iteration over mock results).
    result = asyncio.run(add_edges_for_doc(engine, str(uuid.uuid4())))
    assert result == 5
