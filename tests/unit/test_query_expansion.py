"""Tests for dewie.query.expand — query expansion with generic synonyms."""

from dewie.query.expand import expand_query


def test_expands_known_synonyms():
    result = expand_query("install guide")
    # original preserved
    assert result.startswith("install guide")
    # synonyms appended
    assert "installation" in result
    assert "tutorial" in result


def test_no_synonyms_returns_original():
    result = expand_query("how to bake bread")
    assert result == "how to bake bread"


def test_empty_query_returned_unchanged():
    assert expand_query("") == ""
    assert expand_query("   ") == "   "


def test_no_duplicate_terms_added():
    # "setup" is a synonym of both "config" and "install"; must appear once
    result = expand_query("config install")
    assert result.split().count("setup") == 1


def test_skips_terms_already_present():
    # "document" expands to include "file"; if query already has it, don't re-add
    result = expand_query("document file")
    assert result.split().count("file") == 1
