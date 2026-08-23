"""
Tests for deterministic document ID generation (uuid5 + URL normalisation).
"""

from dewie.models.content import ContentDocument, make_doc_id


def test_make_doc_id_is_deterministic():
    url = "https://example.com/article"
    assert make_doc_id(url) == make_doc_id(url)


def test_make_doc_id_strips_utm():
    clean = "https://example.com/article"
    dirty = "https://example.com/article?utm_source=twitter&utm_medium=social"
    assert make_doc_id(dirty) == make_doc_id(clean)


def test_make_doc_id_strips_trailing_slash():
    assert make_doc_id("https://example.com/article/") == make_doc_id("https://example.com/article")


def test_make_doc_id_strips_www():
    assert make_doc_id("https://www.example.com/article") == make_doc_id(
        "https://example.com/article"
    )


def test_different_urls_different_ids():
    assert make_doc_id("https://example.com/a") != make_doc_id("https://example.com/b")


def test_content_document_from_url():
    url = "https://example.com/test-article"
    doc = ContentDocument.from_url(url, title="test")
    assert doc.id == make_doc_id(url)
    assert doc.url == url
    assert doc.title == "test"
