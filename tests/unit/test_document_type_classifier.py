"""Unit tests for document type / author / tone / reading_level classification (issue #7)."""

from dewie.enrichment.schema import build_extraction_prompt
from dewie.models.content import ContentDocument, DocumentType, ReadingLevel


def test_reading_level_enum_values():
    assert ReadingLevel.QUICK_READ == "quick_read"
    assert ReadingLevel.STANDARD == "standard"
    assert ReadingLevel.LONG_READ == "long_read"
    assert ReadingLevel.DEEP_DIVE == "deep_dive"
    assert ReadingLevel.ACADEMIC == "academic"


def test_document_type_enum_values():
    values = {e.value for e in DocumentType}
    assert "news_article" in values
    assert "blog_post" in values
    assert "academic_paper" in values
    assert "forum_post" in values
    assert "social_media" in values
    assert "documentation" in values
    assert "video" in values
    assert "podcast" in values
    assert "other" in values


def test_content_document_has_author_field():
    doc = ContentDocument(url="https://x.com", author="Jane Smith")
    assert doc.author == "Jane Smith"


def test_content_document_has_reading_level_field():
    doc = ContentDocument(url="https://x.com", reading_level=ReadingLevel.ACADEMIC)
    assert doc.reading_level == ReadingLevel.ACADEMIC
    assert doc.reading_level == "academic"


def test_content_document_defaults_are_none():
    doc = ContentDocument(url="https://x.com")
    assert doc.document_type is None
    assert doc.author is None
    assert doc.reading_level is None


def test_extraction_prompt_contains_reading_level():
    prompt = build_extraction_prompt("Test title", "Test body")
    assert "reading_level" in prompt


def test_extraction_prompt_contains_author():
    prompt = build_extraction_prompt("t", "b")
    assert "author" in prompt


def test_extraction_prompt_contains_document_type():
    prompt = build_extraction_prompt("t", "b")
    assert "document_type" in prompt


def test_row_to_doc_unknown_document_type_returns_none():
    """_row_to_doc must not crash on an unrecognized document_type string."""
    from dewie.models.content import DocumentType
    from dewie.storage.postgres import _safe_enum

    result = _safe_enum(DocumentType, "unknown_future_type")
    assert result is None


def test_row_to_doc_unknown_reading_level_returns_none():
    """_row_to_doc must not crash on an unrecognized reading_level string."""
    from dewie.models.content import ReadingLevel
    from dewie.storage.postgres import _safe_enum

    result = _safe_enum(ReadingLevel, "ultra_deep_dive")
    assert result is None


def test_row_to_doc_valid_enums_coerce_correctly():
    """Known enum values must coerce cleanly."""
    from dewie.models.content import DocumentType, ReadingLevel
    from dewie.storage.postgres import _safe_enum

    assert _safe_enum(DocumentType, "news_article") == DocumentType.NEWS_ARTICLE
    assert _safe_enum(ReadingLevel, "academic") == ReadingLevel.ACADEMIC


def test_safe_enum_null_returns_none():
    from dewie.models.content import DocumentType
    from dewie.storage.postgres import _safe_enum

    assert _safe_enum(DocumentType, None) is None
