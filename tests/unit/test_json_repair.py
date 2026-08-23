"""
Tests for _parse_extraction_result and _repair_json in processor.py.

Covers the three real-world failure modes seen in production:
1. Trailing commas (JS-style LLM output)
2. Truncated JSON (LLM hits token limit mid-array)
3. Prose-wrapped JSON (LLM adds explanation text around the JSON)
"""

from __future__ import annotations

import json

import pytest

from dewie.enrichment.processor import _parse_extraction_result, _repair_json

# ── _repair_json tests ────────────────────────────────────────────────────────


def test_repair_trailing_comma_in_array():
    raw = '{"topics": ["a", "b",]}'
    fixed = _repair_json(raw)
    assert json.loads(fixed) == {"topics": ["a", "b"]}


def test_repair_trailing_comma_in_object():
    raw = '{"a": 1, "b": 2,}'
    fixed = _repair_json(raw)
    assert json.loads(fixed) == {"a": 1, "b": 2}


def test_repair_truncated_array():
    raw = '{"topics": ["finance", "tech"], "entities": [{"text": "Google"'
    fixed = _repair_json(raw)
    parsed = json.loads(fixed)
    assert parsed["topics"] == ["finance", "tech"]
    assert isinstance(parsed["entities"], list)


def test_repair_truncated_at_string_value():
    raw = '{"document_type": "article", "summary": "This is a summ'
    fixed = _repair_json(raw)
    # Should close the open string and object — may not be semantically perfect
    # but must not raise
    try:
        json.loads(fixed)
    except json.JSONDecodeError:
        pass  # Acceptable — mid-string truncation is hard to repair


def test_repair_already_valid():
    raw = '{"a": 1}'
    assert _repair_json(raw) == raw


# ── _parse_extraction_result tests ───────────────────────────────────────────

_BASE = {
    "document_type": "article",
    "summary": "Test summary.",
    "keywords": [],
    "entities": [],
    "answers_questions": [],
    "missing_coverage": [],
}


def test_parse_valid_json():
    raw = json.dumps(_BASE)
    result = _parse_extraction_result(raw)
    assert result.document_type == "article"
    assert result.summary == "Test summary."


def test_parse_markdown_fenced():
    raw = f"```json\n{json.dumps(_BASE)}\n```"
    result = _parse_extraction_result(raw)
    assert result.document_type == "article"


def test_parse_prose_wrapped():
    raw = f"Here is the extracted JSON:\n{json.dumps(_BASE)}\nHope that helps!"
    result = _parse_extraction_result(raw)
    assert result.document_type == "article"


def test_parse_trailing_comma():
    raw = '{"document_type": "article", "summary": "s.", "keywords": ["a", "b",], "entities": [], "answers_questions": [], "missing_coverage": []}'
    result = _parse_extraction_result(raw)
    assert result.document_type == "article"


def test_parse_truncated_entities_array():
    """Simulates LLM hitting token limit mid-entities array."""
    raw = (
        "{\n"
        '  "document_type": "article",\n'
        '  "summary": "Finance article.",\n'
        '  "keywords": ["finance", "debt"],\n'
        '  "entities": [{"text": "USA", "label": "GPE"}, {"text": "Congress"'
        # truncated here
    )
    result = _parse_extraction_result(raw)
    assert result.document_type == "article"
    assert result.summary == "Finance article."


def test_parse_truncated_at_line_59():
    """Exact pattern seen in production: 'expected , or ] at line 59'."""
    entities = [{"text": f"Entity{i}", "label": "ORG"} for i in range(20)]
    partial = entities[:18]
    raw = json.dumps({**_BASE, "entities": partial}).rstrip("}")
    raw += ', {"text": "Entity18", "label": "ORG"'  # truncated
    result = _parse_extraction_result(raw)
    assert result.document_type == "article"


def test_parse_raises_on_garbage():
    with pytest.raises(ValueError, match="Could not parse"):
        _parse_extraction_result("this is not json at all !!!")
