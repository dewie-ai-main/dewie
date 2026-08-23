# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""Tests for _doc_type document type classification.

Covers issue #931: type tabs not filtering correctly due to inconsistent
classification between frontend (docType) and backend (_doc_type).
"""

import pytest

from dewie.api.routes.documents import _doc_type


class FakeDoc:
    """Minimal fake document with url and source attributes."""
    def __init__(self, url: str, source: str = ""):
        self.url = url
        self.source = source


# ── YouTube ──────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("url,expected", [
    ("https://www.youtube.com/watch?v=abc123", "youtube"),
    ("https://youtu.be/abc123", "youtube"),
    ("https://YOUTUBE.COM/watch?v=xyz", "youtube"),
    ("https://www.youtu.be/xyz", "youtube"),
    ("https://www.youtube.com/embed/abc", "youtube"),
    ("https://example.com/page", "website"),
    ("https://en.wikipedia.org/wiki/Test", "website"),
])
def test_youtube_detection(url, expected):
    doc = FakeDoc(url)
    assert _doc_type(doc) == expected


# ── Podcast (source-based) ──────────────────────────────────────────────────

@pytest.mark.parametrize("url,source,expected", [
    ("https://example.com/episode", "podcast", "podcast"),
    ("https://example.com/episode", "podcasts", "podcast"),
    ("https://en.wikipedia.org/wiki/Music", "podcast", "podcast"),
    ("https://en.wikipedia.org/wiki/Test", "upload", "document"),
    ("https://example.com/page", "podcaster", "website"),
])
def test_podcast_source_detection(url, source, expected):
    doc = FakeDoc(url, source=source)
    assert _doc_type(doc) == expected


# ── Podcast (extension-based in URL path) ────────────────────────────────────

@pytest.mark.parametrize("url,expected", [
    ("https://example.com/audio.mp3", "podcast"),
    ("https://example.com/show/ep1.m4a", "podcast"),
    ("https://example.com/podcast/ep2.wav", "podcast"),
    ("https://example.com/ep3.ogg", "podcast"),
    ("https://example.com/audio.mp4", "podcast"),
    ("https://example.com/file.podcast", "podcast"),
    ("https://EXAMPLE.COM/song.MP3", "podcast"),
])
def test_podcast_extension_detection(url, expected):
    doc = FakeDoc(url)
    assert _doc_type(doc) == expected


# ── PDF detection ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("url,expected", [
    ("https://example.com/doc.pdf", "pdf"),
    ("https://example.com/path/document.PDF", "pdf"),
    ("https://en.wikipedia.org/wiki/Sample.pdf", "pdf"),
])
def test_pdf_detection(url, expected):
    doc = FakeDoc(url)
    assert _doc_type(doc) == expected


# ── Document detection (source-based) ───────────────────────────────────────

@pytest.mark.parametrize("url,expected", [
    ("https://example.com/file.docx", "document"),
    ("https://example.com/file.doc", "document"),
    ("https://example.com/file.xlsx", "document"),
    ("https://example.com/file.xls", "document"),
    ("https://example.com/file.pptx", "document"),
    ("https://example.com/file.ppt", "document"),
    ("https://example.com/file.odt", "document"),
    ("https://example.com/file.ods", "document"),
    ("https://example.com/file.odp", "document"),
    ("https://example.com/file.pages", "document"),
    ("https://example.com/file.numbers", "document"),
    ("https://example.com/file.key", "document"),
])
def test_document_extension_detection(url, expected):
    doc = FakeDoc(url)
    assert _doc_type(doc) == expected


# ── Query parameter false positive guards (issue #931) ────────────────────────

@pytest.mark.parametrize("url,expected", [
    ("https://en.wikipedia.org/wiki/Some_topic", "website"),
    ("https://en.wikipedia.org/wiki/Music", "website"),
    ("https://en.wikipedia.org/wiki/Radio", "website"),
    ("https://en.wikipedia.org/wiki/Podcast", "website"),
    ("https://en.wikipedia.org/wiki/YouTube", "website"),
    ("https://en.wikipedia.org/wiki/File:audio_example", "website"),
    ("https://example.com/page?file=mp3", "website"),
    ("https://example.com/search?q=mp3", "website"),
    ("https://example.com/search?q=m4a", "website"),
    ("https://example.com/search?q=podcast", "website"),
    ("https://example.com/page?query=docx", "website"),
    ("https://example.com/page?query=pdf", "website"),
    ("https://en.wikipedia.org/wiki/MP3", "website"),
    ("https://en.wikipedia.org/wiki/MPEG-4", "website"),
    ("https://en.wikipedia.org/wiki/AAC_audio", "website"),
    ("https://en.wikipedia.org/wiki/WAV", "website"),
    ("https://en.wikipedia.org/wiki/OGG", "website"),
    ("https://example.com/path/something/mp3/file", "website"),
    ("https://example.com/path/something/m4a/file", "website"),
    ("https://example.com/path/something/podcast/file", "website"),
])
def test_no_false_positives_from_query_params(url, expected):
    """URLs that contain audio/file terms in query params should NOT be
    misclassified as podcasts, pdfs, or documents (issue #931)."""
    doc = FakeDoc(url)
    assert _doc_type(doc) == expected, f"URL '{url}' was classified as '{_doc_type(doc)}' instead of '{expected}'"


# ── Path segments that look like extensions (issue #931) ────────────────────

@pytest.mark.parametrize("url,expected", [
    ("https://example.com/category/podcast/episode", "website"),
    ("https://example.com/episode/podcast", "website"),
    ("https://en.wikipedia.org/wiki/File:Audio_example", "website"),
    ("https://example.com/path/to/mp3", "website"),
    ("https://example.com/path/to/m4a", "website"),
])
def test_no_false_positives_from_path_segments(url, expected):
    """URL path segments that look like file extensions but aren't actual
    file names should NOT be misclassified (issue #931)."""
    doc = FakeDoc(url)
    assert _doc_type(doc) == expected, f"URL '{url}' was classified as '{_doc_type(doc)}' instead of '{expected}'"


# ── Catch-all / website ─────────────────────────────────────────────────────

@pytest.mark.parametrize("url", [
    "https://example.com/article",
    "https://en.wikipedia.org/wiki/Main_Page",
    "https://github.com/repo/issues/931",
    "https://news.ycombinator.com/",
    "https://blog.example.com/post",
])
def test_default_website(url):
    doc = FakeDoc(url)
    assert _doc_type(doc) == "website"
