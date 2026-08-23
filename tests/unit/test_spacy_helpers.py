"""Tests for pure helper functions in dewie.enrichment.backends.spacy."""

from __future__ import annotations

import pytest

pytest.importorskip("spacy", reason="requires spacy — optional NLP backend")

# ── _extract_summary ──────────────────────────────────────────────────────────


def test_extract_summary_empty():
    from dewie.enrichment.backends.spacy import _extract_summary

    assert _extract_summary("") == ""


def test_extract_summary_short():
    from dewie.enrichment.backends.spacy import _extract_summary

    body = "This is a short sentence."
    assert _extract_summary(body) == body


def test_extract_summary_truncates_at_max_chars():
    from dewie.enrichment.backends.spacy import _extract_summary

    body = "A" * 600
    result = _extract_summary(body, max_chars=500)
    assert len(result) <= 500


def test_extract_summary_multiple_sentences():
    from dewie.enrichment.backends.spacy import _extract_summary

    body = "First sentence. Second sentence. Third sentence."
    result = _extract_summary(body, max_chars=500)
    assert "First sentence" in result


def test_extract_summary_stops_at_limit():
    from dewie.enrichment.backends.spacy import _extract_summary

    body = "Short. " + ("A very long sentence that goes on and on. " * 20)
    result = _extract_summary(body, max_chars=50)
    assert len(result) <= 50 or result == "Short."


# ── _compute_sentiment ────────────────────────────────────────────────────────


def _make_mock_doc(tokens: list[str]):
    """Create a minimal mock spaCy-like doc with alpha tokens."""

    class MockToken:
        def __init__(self, text):
            self.text = text
            self.is_alpha = text.isalpha()

    class MockDoc:
        def __iter__(self):
            return iter([MockToken(t) for t in tokens])

    return MockDoc()


def test_compute_sentiment_no_tokens():
    from dewie.enrichment.backends.spacy import _compute_sentiment

    doc = _make_mock_doc([])
    assert _compute_sentiment(doc) == pytest.approx(0.0)


def test_compute_sentiment_positive():
    from dewie.enrichment.backends.spacy import _compute_sentiment

    doc = _make_mock_doc(["good", "great", "excellent"])
    score = _compute_sentiment(doc)
    assert score > 0


def test_compute_sentiment_negative():
    from dewie.enrichment.backends.spacy import _compute_sentiment

    doc = _make_mock_doc(["bad", "terrible", "crisis"])
    score = _compute_sentiment(doc)
    assert score < 0


def test_compute_sentiment_neutral():
    from dewie.enrichment.backends.spacy import _compute_sentiment

    doc = _make_mock_doc(["the", "cat", "sat"])
    assert _compute_sentiment(doc) == pytest.approx(0.0)


def test_compute_sentiment_mixed():
    from dewie.enrichment.backends.spacy import _compute_sentiment

    doc = _make_mock_doc(["good", "bad"])
    assert _compute_sentiment(doc) == pytest.approx(0.0)


# ── SpacyBackend basic properties ─────────────────────────────────────────────


def test_spacy_backend_name():
    from dewie.enrichment.backends.spacy import SpacyBackend

    b = SpacyBackend()
    assert b.name == "spacy"


def test_spacy_backend_set_document():
    from dewie.enrichment.backends.spacy import SpacyBackend

    b = SpacyBackend()
    b._set_document("Title", "Body text here.")
    assert b._title == "Title"
    assert b._body == "Body text here."


def test_spacy_backend_defaults():
    from dewie.enrichment.backends.spacy import SpacyBackend

    b = SpacyBackend()
    assert b._max_keywords == 15
    assert b._max_topics == 8
    assert b._max_entities == 20


def test_spacy_backend_custom_params():
    from dewie.enrichment.backends.spacy import SpacyBackend

    b = SpacyBackend(max_keywords=5, max_topics=3, max_entities=10)
    assert b._max_keywords == 5


# ── Constants ─────────────────────────────────────────────────────────────────


def test_entity_labels_not_empty():
    from dewie.enrichment.backends.spacy import _ENTITY_LABELS

    assert "ORG" in _ENTITY_LABELS
    assert "PERSON" in _ENTITY_LABELS


def test_pos_neg_words_not_empty():
    from dewie.enrichment.backends.spacy import _NEG_WORDS, _POS_WORDS

    assert len(_POS_WORDS) > 0
    assert len(_NEG_WORDS) > 0
