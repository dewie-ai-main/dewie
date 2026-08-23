"""Tests for dewie.pipeline — pure helpers (no DB/HTTP calls)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from dewie.pipeline import (
    build_embed_text,
    compute_quality_score,
    embed_batch,
    extract_ke,
    generate_aq,
    jaccard,
    tokenize,
)

# ── jaccard ───────────────────────────────────────────────────────────────────


def test_jaccard_identical():
    assert jaccard({"a", "b", "c"}, {"a", "b", "c"}) == pytest.approx(1.0)


def test_jaccard_disjoint():
    assert jaccard({"a"}, {"b"}) == pytest.approx(0.0)


def test_jaccard_partial():
    score = jaccard({"a", "b"}, {"b", "c"})
    assert 0.0 < score < 1.0


def test_jaccard_empty_a():
    assert jaccard(set(), {"a", "b"}) == 0.0


def test_jaccard_empty_b():
    assert jaccard({"a"}, set()) == 0.0


# ── tokenize ──────────────────────────────────────────────────────────────────


def test_tokenize_basic():
    tokens = tokenize(["machine learning", "neural-networks"])
    assert "machine" in tokens
    assert "learning" in tokens
    assert "neural" in tokens


def test_tokenize_lowercases():
    tokens = tokenize(["Machine Learning"])
    assert "machine" in tokens
    assert "Machine" not in tokens


def test_tokenize_filters_short():
    tokens = tokenize(["a", "ab", "abc"])
    assert "a" not in tokens
    assert "ab" not in tokens
    assert "abc" in tokens


def test_tokenize_empty():
    assert tokenize([]) == set()


def test_tokenize_ignores_non_strings():
    tokens = tokenize(["valid", None, 123])
    assert "valid" in tokens


def test_tokenize_splits_on_punctuation():
    tokens = tokenize(["foo/bar.baz,qux"])
    assert "foo" in tokens
    assert "bar" in tokens


# ── build_embed_text ──────────────────────────────────────────────────────────


def test_build_embed_text_basic():
    text = build_embed_text("My Title", "My summary.", [], "")
    assert "Title: My Title" in text
    assert "Summary: My summary." in text


def test_build_embed_text_with_aq():
    text = build_embed_text("Title", "Summary", ["Q1?", "Q2?", "Q3?"], "body")
    assert "Questions:" in text
    assert "Q1?" in text


def test_build_embed_text_uses_embed_summary_over_summary():
    text = build_embed_text("Title", "display summary", [], "body", embed_summary="dense summary")
    assert "dense summary" in text
    assert "display summary" not in text


def test_build_embed_text_no_embed_summary_falls_back():
    text = build_embed_text("Title", "fallback summary", [], "body", embed_summary="")
    assert "fallback summary" in text


def test_build_embed_text_truncates_aq_to_6():
    aq = ["Q1?", "Q2?", "Q3?", "Q4?", "Q5?", "Q6?", "Q7?", "Q8?"]
    text = build_embed_text("T", "S", aq, "body")
    assert "Q7?" not in text
    assert "Q6?" in text


# ── compute_quality_score ─────────────────────────────────────────────────────


def test_quality_score_perfect_doc():
    body = "x" * 3000
    keywords = ["a"] * 15
    entities = ["E"] * 8
    aq = ["Q?"] * 6
    summary = "x" * 200
    score = compute_quality_score(body, keywords, entities, aq, summary)
    assert score == 100


def test_quality_score_empty_doc():
    score = compute_quality_score("", [], [], [], "")
    assert score == 0


def test_quality_score_short_body():
    body = "x" * 100
    score = compute_quality_score(body, [], [], [], "")
    assert score == 0


def test_quality_score_medium_body():
    body = "x" * 500
    score = compute_quality_score(body, [], [], [], "")
    assert score == 16


def test_quality_score_long_body():
    body = "x" * 1500
    score = compute_quality_score(body, [], [], [], "")
    assert score == 26


def test_quality_score_capped_at_100():
    body = "x" * 5000
    keywords = ["k"] * 30
    entities = ["E"] * 15
    aq = ["Q?"] * 10
    summary = "x" * 500
    score = compute_quality_score(body, keywords, entities, aq, summary)
    assert score == 100


def test_quality_score_few_keywords():
    body = "x" * 3000
    score_no_kw = compute_quality_score(body, [], [], [], "")
    score_few_kw = compute_quality_score(body, ["k"], [], [], "")
    assert score_few_kw > score_no_kw


def test_quality_score_entities_contribution():
    body = "x" * 3000
    score_no_ent = compute_quality_score(body, [], [], [], "")
    score_ent = compute_quality_score(body, [], ["E1", "E2"], [], "")
    assert score_ent > score_no_ent


def test_quality_score_aq_contribution():
    body = "x" * 3000
    score_no_aq = compute_quality_score(body, [], [], [], "")
    score_aq = compute_quality_score(body, [], [], ["Q1?", "Q2?"], "")
    assert score_aq > score_no_aq


def test_quality_score_summary_contribution():
    body = "x" * 3000
    score_no_sum = compute_quality_score(body, [], [], [], "")
    score_sum = compute_quality_score(body, [], [], [], "x" * 200)
    assert score_sum > score_no_sum


def test_quality_score_none_inputs():
    score = compute_quality_score(None, None, None, None, None)
    assert score == 0


# ── generate_aq (async, LLM call) ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_generate_aq_returns_list():
    mock_provider = AsyncMock()
    mock_provider.complete = AsyncMock(return_value='["What is X?", "How does X work?"]')

    with patch("dewie.pipeline.get_chat_provider", return_value=mock_provider):
        result = await generate_aq(None, "Title", "Summary", "Content")

    assert isinstance(result, list)
    assert "What is X?" in result


@pytest.mark.asyncio
async def test_generate_aq_returns_empty_on_none_response():
    mock_provider = AsyncMock()
    mock_provider.complete = AsyncMock(return_value=None)

    with patch("dewie.pipeline.get_chat_provider", return_value=mock_provider):
        result = await generate_aq(None, "Title", "Summary", "")

    assert result == []


@pytest.mark.asyncio
async def test_generate_aq_returns_empty_on_exception():
    mock_provider = AsyncMock()
    mock_provider.complete = AsyncMock(side_effect=Exception("LLM error"))

    with patch("dewie.pipeline.get_chat_provider", return_value=mock_provider):
        result = await generate_aq(None, "Title", "Summary", "Content")

    assert result == []


@pytest.mark.asyncio
async def test_generate_aq_strips_code_fences():
    mock_provider = AsyncMock()
    mock_provider.complete = AsyncMock(return_value='```json\n["Q1?", "Q2?"]\n```')

    with patch("dewie.pipeline.get_chat_provider", return_value=mock_provider):
        result = await generate_aq(None, "T", "S", "C")

    assert "Q1?" in result


@pytest.mark.asyncio
async def test_generate_aq_truncates_to_8():
    mock_provider = AsyncMock()
    qs = [f"Q{i}?" for i in range(12)]
    mock_provider.complete = AsyncMock(return_value=str(qs).replace("'", '"'))

    with patch("dewie.pipeline.get_chat_provider", return_value=mock_provider):
        result = await generate_aq(None, "T", "S", "C")

    assert len(result) <= 8


# ── extract_ke (async, LLM call) ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_extract_ke_returns_dict():
    mock_provider = AsyncMock()
    mock_provider.complete = AsyncMock(
        return_value='{"keywords": ["ml", "ai"], "entities": ["OpenAI"]}'
    )

    with patch("dewie.pipeline.get_chat_provider", return_value=mock_provider):
        result = await extract_ke(None, "Title", "Summary", "Content")

    assert "keywords" in result
    assert "entities" in result
    assert "ml" in result["keywords"]
    assert "OpenAI" in result["entities"]


@pytest.mark.asyncio
async def test_extract_ke_returns_empty_on_none():
    mock_provider = AsyncMock()
    mock_provider.complete = AsyncMock(return_value=None)

    with patch("dewie.pipeline.get_chat_provider", return_value=mock_provider):
        result = await extract_ke(None, "T", "S", "C")

    assert result == {"keywords": [], "entities": []}


@pytest.mark.asyncio
async def test_extract_ke_returns_empty_on_exception():
    mock_provider = AsyncMock()
    mock_provider.complete = AsyncMock(side_effect=RuntimeError("oops"))

    with patch("dewie.pipeline.get_chat_provider", return_value=mock_provider):
        result = await extract_ke(None, "T", "S", "C")

    assert result == {"keywords": [], "entities": []}


@pytest.mark.asyncio
async def test_extract_ke_lowercases_keywords():
    mock_provider = AsyncMock()
    mock_provider.complete = AsyncMock(
        return_value='{"keywords": ["Machine Learning"], "entities": []}'
    )

    with patch("dewie.pipeline.get_chat_provider", return_value=mock_provider):
        result = await extract_ke(None, "T", "S", "C")

    assert "machine learning" in result["keywords"]


# ── embed_batch ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_embed_batch_returns_vectors():
    mock_provider = AsyncMock()
    mock_provider.embed = AsyncMock(return_value=[[0.1, 0.2], [0.3, 0.4]])

    with patch("dewie.pipeline.get_embedding_provider", return_value=mock_provider):
        result = await embed_batch(None, ["text1", "text2"])

    assert result == [[0.1, 0.2], [0.3, 0.4]]


@pytest.mark.asyncio
async def test_embed_batch_returns_none_on_failure():
    mock_provider = AsyncMock()
    mock_provider.embed = AsyncMock(return_value=None)

    with patch("dewie.pipeline.get_embedding_provider", return_value=mock_provider):
        result = await embed_batch(None, ["text"])

    assert result is None
