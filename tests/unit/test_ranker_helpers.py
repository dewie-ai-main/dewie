"""Tests for pure helper functions in dewie.storage.rankers."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

# ── _cosine ───────────────────────────────────────────────────────────────────


def test_cosine_identical():
    from dewie.storage.rankers import _cosine

    v = [1.0, 0.0, 0.0]
    assert _cosine(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal():
    from dewie.storage.rankers import _cosine

    assert _cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_zero_vector():
    from dewie.storage.rankers import _cosine

    assert _cosine([0.0, 0.0], [1.0, 2.0]) == pytest.approx(0.0)


def test_cosine_opposite():
    from dewie.storage.rankers import _cosine

    assert _cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_partial():
    from dewie.storage.rankers import _cosine

    a = [1.0, 1.0]
    b = [1.0, 0.0]
    result = _cosine(a, b)
    assert 0 < result < 1


# ── _mmr_rerank ───────────────────────────────────────────────────────────────


def test_mmr_rerank_no_embeddings_returns_candidates():
    from dewie.storage.rankers import _mmr_rerank

    candidates = [("d1", 0.9), ("d2", 0.8), ("d3", 0.7)]
    result = _mmr_rerank(candidates, {}, None, limit=3)
    assert len(result) <= 3


def test_mmr_rerank_with_embeddings():
    from dewie.storage.rankers import _mmr_rerank

    candidates = [("d1", 0.9), ("d2", 0.8)]
    embeddings = {
        "d1": [1.0, 0.0],
        "d2": [0.0, 1.0],
    }
    result = _mmr_rerank(candidates, embeddings, [1.0, 0.0], limit=2)
    assert len(result) == 2
    assert result[0][0] == "d1"  # most relevant first


def test_mmr_rerank_limit():
    from dewie.storage.rankers import _mmr_rerank

    candidates = [("d1", 0.9), ("d2", 0.8), ("d3", 0.7)]
    embeddings = {"d1": [1.0, 0.0], "d2": [0.9, 0.1], "d3": [0.1, 0.9]}
    result = _mmr_rerank(candidates, embeddings, [1.0, 0.0], limit=2)
    assert len(result) <= 2


def test_mmr_rerank_empty_candidates():
    from dewie.storage.rankers import _mmr_rerank

    result = _mmr_rerank([], {}, None, limit=5)
    assert result == []


# ── apply_staleness_penalty ───────────────────────────────────────────────────


def _make_result(score: float, ingested_at):
    class FakeResult:
        pass

    r = FakeResult()
    r.score = score
    r.ingested_at = ingested_at
    return r


def test_staleness_no_penalty_fresh():
    from dewie.storage.rankers import apply_staleness_penalty

    now = datetime.now(UTC)
    r = _make_result(1.0, now - timedelta(days=3))
    apply_staleness_penalty([r])
    assert r.score == pytest.approx(1.0)


def test_staleness_slight_penalty_week():
    from dewie.storage.rankers import apply_staleness_penalty

    now = datetime.now(UTC)
    r = _make_result(1.0, now - timedelta(days=15))
    apply_staleness_penalty([r])
    assert r.score == pytest.approx(0.97)


def test_staleness_medium_penalty_month():
    from dewie.storage.rankers import apply_staleness_penalty

    now = datetime.now(UTC)
    r = _make_result(1.0, now - timedelta(days=60))
    apply_staleness_penalty([r])
    assert r.score == pytest.approx(0.93)


def test_staleness_higher_penalty_quarter():
    from dewie.storage.rankers import apply_staleness_penalty

    now = datetime.now(UTC)
    r = _make_result(1.0, now - timedelta(days=200))
    apply_staleness_penalty([r])
    assert r.score == pytest.approx(0.88)


def test_staleness_max_penalty_old():
    from dewie.storage.rankers import apply_staleness_penalty

    now = datetime.now(UTC)
    r = _make_result(1.0, now - timedelta(days=500))
    apply_staleness_penalty([r])
    assert r.score == pytest.approx(0.80)


def test_staleness_none_ingested_at_no_change():
    from dewie.storage.rankers import apply_staleness_penalty

    r = _make_result(0.9, None)
    apply_staleness_penalty([r])
    assert r.score == pytest.approx(0.9)


def test_staleness_naive_datetime_handled():
    from dewie.storage.rankers import apply_staleness_penalty

    naive = datetime.utcnow() - timedelta(days=3)  # no tzinfo
    r = _make_result(1.0, naive)
    apply_staleness_penalty([r])
    assert r.score == pytest.approx(1.0)


# ── _corpus_filter_sql (removed in Dewie rename — skipped) ───────────────────


@pytest.mark.skip(reason="_corpus_filter_sql removed from rankers.py")
def test_corpus_filter_all():
    pass


@pytest.mark.skip(reason="_corpus_filter_sql removed from rankers.py")
def test_corpus_filter_public():
    pass


@pytest.mark.skip(reason="_corpus_filter_sql removed from rankers.py")
def test_corpus_filter_personal():
    pass


@pytest.mark.skip(reason="_corpus_filter_sql removed from rankers.py")
def test_corpus_filter_personal_empty_ids():
    pass


# ── _extract_query_terms ──────────────────────────────────────────────────────


def test_extract_query_terms_proper_nouns():
    from dewie.storage.rankers import _extract_query_terms

    terms = _extract_query_terms("Apple reported strong earnings")
    assert "Apple" in terms


def test_extract_query_terms_tickers():
    from dewie.storage.rankers import _extract_query_terms

    terms = _extract_query_terms("AAPL hit a new high today")
    assert "AAPL" in terms


def test_extract_query_terms_excludes_stop_words():
    from dewie.storage.rankers import _extract_query_terms

    terms = _extract_query_terms("what is the score today")
    assert "what" not in terms
    assert "score" not in terms


def test_extract_query_terms_long_words():
    from dewie.storage.rankers import _extract_query_terms

    terms = _extract_query_terms("machine learning algorithms")
    assert "machine" in terms
    assert "learning" in terms
    assert "algorithms" in terms


def test_extract_query_terms_empty():
    from dewie.storage.rankers import _extract_query_terms

    terms = _extract_query_terms("")
    assert terms == []


# ── list_rankers ──────────────────────────────────────────────────────────────


def test_list_rankers_returns_list():
    from dewie.storage.rankers import list_rankers

    rankers = list_rankers()
    assert isinstance(rankers, list)
    assert len(rankers) > 0


def test_list_rankers_have_required_keys():
    from dewie.storage.rankers import list_rankers

    for r in list_rankers():
        assert "label" in r
        assert "description" in r
