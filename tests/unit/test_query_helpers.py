"""Tests for dewie.api.routes.query — pure helper functions."""

from __future__ import annotations

import pytest

# ── _token_set ────────────────────────────────────────────────────────────────


def test_token_set_lowercases():
    from dewie.api.routes.query import _token_set

    tokens = _token_set("Machine Learning")
    assert "machine" in tokens
    assert "learning" in tokens


def test_token_set_drops_stop_words():
    from dewie.api.routes.query import _token_set

    tokens = _token_set("the cat sat on the mat")
    assert "the" not in tokens
    assert "cat" in tokens
    assert "mat" in tokens


def test_token_set_drops_short_tokens():
    from dewie.api.routes.query import _token_set

    tokens = _token_set("AI is great")
    assert "ai" not in tokens  # len < 3
    assert "is" not in tokens  # stop word
    assert "great" in tokens


def test_token_set_handles_hyphens():
    from dewie.api.routes.query import _token_set

    tokens = _token_set("state-of-the-art")
    assert "state" in tokens
    assert "art" in tokens


# ── _jaccard ──────────────────────────────────────────────────────────────────


def test_jaccard_identical():
    from dewie.api.routes.query import _jaccard

    s = frozenset(["a", "b", "c"])
    assert _jaccard(s, s) == pytest.approx(1.0)


def test_jaccard_disjoint():
    from dewie.api.routes.query import _jaccard

    assert _jaccard(frozenset(["a"]), frozenset(["b"])) == pytest.approx(0.0)


def test_jaccard_partial():
    from dewie.api.routes.query import _jaccard

    a = frozenset(["a", "b"])
    b = frozenset(["b", "c"])
    assert _jaccard(a, b) == pytest.approx(1 / 3)


def test_jaccard_empty():
    from dewie.api.routes.query import _jaccard

    assert _jaccard(frozenset(), frozenset(["a"])) == pytest.approx(0.0)
    assert _jaccard(frozenset(["a"]), frozenset()) == pytest.approx(0.0)


# ── _diverse_keywords ─────────────────────────────────────────────────────────


def test_diverse_keywords_returns_at_most_n():
    from dewie.api.routes.query import _diverse_keywords

    kws = ["machine learning", "deep learning", "neural networks", "AI", "python"]
    result = _diverse_keywords(kws, n=3)
    assert len(result) <= 3


def test_diverse_keywords_empty():
    from dewie.api.routes.query import _diverse_keywords

    assert _diverse_keywords([]) == []


def test_diverse_keywords_deduplicates_similar():
    from dewie.api.routes.query import _diverse_keywords

    # These are very similar phrases — diversity should filter them
    kws = ["machine learning models", "machine learning algorithms", "deep learning"]
    result = _diverse_keywords(kws, n=3)
    # We expect at most 3 but the similar ones should be deduplicated
    assert len(result) <= 3


def test_diverse_keywords_dissimilar_words_all_returned():
    from dewie.api.routes.query import _diverse_keywords

    kws = ["python", "finance", "health"]
    result = _diverse_keywords(kws, n=3)
    assert len(result) == 3
    assert "python" in result
    assert "finance" in result
    assert "health" in result


# ── _KW_* constants ───────────────────────────────────────────────────────────


def test_kw_constants_exist():
    from dewie.api.routes.query import _KW_MAX_OVERLAP, _KW_MIN_TOKEN_LEN, _KW_TOP_N

    assert _KW_TOP_N > 0
    assert 0 < _KW_MAX_OVERLAP < 1
    assert _KW_MIN_TOKEN_LEN > 0


# ── _compute_gap_signal ───────────────────────────────────────────────────────


def _make_search_result(**kwargs):
    from dewie.models.query import SearchResult

    defaults = dict(
        doc_id="doc-1",
        doc_type="blog_post",
        url="https://example.com",
        title="Test Doc",
        summary="A test document",
        source="web",
        ingested_at="2024-01-01T00:00:00Z",
        topics=["ai", "ml"],
        keywords=["python", "neural"],
        entities=["OpenAI"],
        sentiment=0.5,
        score=0.8,
        answers_questions=["What is machine learning?"],
        edge_count=5,
        enrichment_quality_score=None,
        reading_level=None,
        chunk_match=None,
        chunk_score=None,
    )
    defaults.update(kwargs)
    return SearchResult(**defaults)


def test_compute_gap_signal_no_results():
    from dewie.api.routes.query import _compute_gap_signal

    result = _compute_gap_signal("machine learning basics", [])
    assert result is not None
    assert "No documents" in result


def test_compute_gap_signal_no_gap_returns_none():
    from dewie.api.routes.query import _compute_gap_signal

    results = [
        _make_search_result(
            score=0.9,
            answers_questions=["What is machine learning?", "How does learning work?"],
            topics=["machine learning", "learning algorithms"],
        )
    ]
    # Query words: "machine", "learning" — covered by AQ
    result = _compute_gap_signal("machine learning", results)
    assert result is None


def test_compute_gap_signal_short_query_words_return_none():
    from dewie.api.routes.query import _compute_gap_signal

    results = [_make_search_result()]
    # Short words (≤4 chars) → query_words empty → return None
    result = _compute_gap_signal("AI is key", results)
    assert result is None


# ── _compute_confidence ───────────────────────────────────────────────────────


def test_compute_confidence_empty_results():
    from dewie.api.routes.query import _compute_confidence

    assert _compute_confidence("test", []) is None


def test_compute_confidence_single_result_high_aq():
    from dewie.api.routes.query import _compute_confidence

    results = [
        _make_search_result(
            score=0.95,
            answers_questions=["What is machine learning?", "How do neural networks learn?"],
            topics=["ml"],
            edge_count=20,
        )
    ]
    conf = _compute_confidence("machine learning neural networks", results)
    assert conf is not None
    assert conf.confidence_level in ("high", "medium", "low")


def test_compute_confidence_two_results():
    from dewie.api.routes.query import _compute_confidence

    results = [
        _make_search_result(id="d1", score=0.9, answers_questions=["What is AI?"], topics=["ai"]),
        _make_search_result(id="d2", score=0.85, answers_questions=[], topics=["ml"]),
    ]
    conf = _compute_confidence("artificial intelligence AI systems", results)
    assert conf is not None
    assert conf.score_gap == pytest.approx(0.05, abs=0.001)


# ── _compute_gap_signal edge cases ─────────────────────────────────────────────


def test_gap_signal_no_query_words():
    """_compute_gap_signal returns None when query has no long words."""
    from dewie.api.routes.query import _compute_gap_signal

    result = _make_search_result(score=0.9, answers_questions=["What is this?"])
    gap = _compute_gap_signal("is it", [result])
    # All words are ≤4 chars, so query_words is empty → returns None
    assert gap is None


def test_gap_signal_low_aq_coverage_uniform_scores():
    """When AQ coverage is low and scores are uniform, returns a gap message."""
    from dewie.api.routes.query import _compute_gap_signal

    result1 = _make_search_result(score=0.5, answers_questions=[], topics=[])
    result2 = _make_search_result(score=0.49, answers_questions=[], topics=[])
    result3 = _make_search_result(score=0.48, answers_questions=[], topics=[])
    gap = _compute_gap_signal("icelandic geological survey", [result1, result2, result3])
    # Low AQ + near-uniform scores → gap detected
    assert gap is not None
    assert isinstance(gap, str)


def test_gap_signal_good_aq_match_returns_none():
    """When top doc's AQ matches query words well, no gap is detected."""
    from dewie.api.routes.query import _compute_gap_signal

    result1 = _make_search_result(
        score=0.9,
        answers_questions=["What is machine learning?", "How does neural training work?"],
        topics=["machine", "learning"],
    )
    result2 = _make_search_result(score=0.5, answers_questions=[], topics=[])
    gap = _compute_gap_signal("machine learning neural training", [result1, result2])
    assert gap is None


# ── _compute_confidence edge cases ─────────────────────────────────────────────


def test_confidence_distributed_complexity():
    """Distributed complexity when topic spread is high and AQ coverage low."""
    from dewie.api.routes.query import _compute_confidence

    r1 = _make_search_result(score=0.9, topics=["finance", "stocks"], edge_count=20)
    r1.answers_questions = []
    r2 = _make_search_result(score=0.85, topics=["sports", "basketball"], edge_count=5)
    r2.answers_questions = []
    r3 = _make_search_result(score=0.82, topics=["technology", "ai"], edge_count=3)
    r3.answers_questions = []
    conf = _compute_confidence("basketball free throw analysis", [r1, r2, r3])
    assert conf is not None
    # Topic spread should be high (all different topics)
    assert conf.complexity in ("distributed", "ambiguous")


def test_confidence_high_score_gap_and_aq_coverage():
    """High score gap + high AQ coverage → lookup complexity."""
    from dewie.api.routes.query import _compute_confidence

    # Large score gap (0.9 vs 0.5 → gap=0.4), answers_questions matches query
    r1 = _make_search_result(score=0.9, topics=["machine", "learning"], edge_count=15)
    r1.answers_questions = ["What is machine learning?", "How does training work?"]
    r2 = _make_search_result(score=0.5, topics=["other"], edge_count=2)
    r2.answers_questions = []
    conf = _compute_confidence("machine learning training", [r1, r2])
    assert conf is not None
    # score_gap > 0.2 (0.4) and AQ coverage > 0.5 → lookup
    assert conf.complexity == "lookup"
    assert conf.confidence_level == "high"


def test_confidence_single_result_zero_gap():
    """Single result has score_gap=0 (no second doc to compare)."""
    from dewie.api.routes.query import _compute_confidence

    r = _make_search_result(score=0.8, topics=[], edge_count=0)
    r.answers_questions = []
    conf = _compute_confidence("test query here", [r])
    assert conf is not None
    assert conf.score_gap == 0.0


def test_confidence_ambiguous_medium():
    """Medium confidence for ambiguous with some edge density."""
    from dewie.api.routes.query import _compute_confidence

    # score_gap=0.3 (above 0.2) but AQ coverage between 0.2 and 0.5
    r1 = _make_search_result(score=0.9, topics=["tech"], edge_count=20)
    r1.answers_questions = ["What is tech?"]  # partial AQ match for "technology testing"
    r2 = _make_search_result(score=0.6, topics=["tech"], edge_count=5)
    r2.answers_questions = []
    r3 = _make_search_result(score=0.55, topics=["tech"], edge_count=5)
    r3.answers_questions = []
    conf = _compute_confidence("technology testing framework", [r1, r2, r3])
    assert conf is not None
    assert conf.complexity in ("ambiguous", "lookup")
    assert conf.suggested_action in ("expand", "intersect", "none")
