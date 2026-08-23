"""Tests for dewie.crawler.extractor — pure link extraction."""

from __future__ import annotations


def test_extract_links_absolute():
    from dewie.crawler.extractor import extract_links

    html = '<a href="https://example.com/page">link</a>'
    links = extract_links(html, "https://example.com")
    assert "https://example.com/page" in links


def test_extract_links_relative():
    from dewie.crawler.extractor import extract_links

    html = '<a href="/about">About</a>'
    links = extract_links(html, "https://example.com")
    assert "https://example.com/about" in links


def test_extract_links_same_domain_filter():
    from dewie.crawler.extractor import extract_links

    html = '<a href="https://other.com/page">External</a>'
    links = extract_links(html, "https://example.com", same_domain=True)
    assert len(links) == 0


def test_extract_links_allow_cross_domain():
    from dewie.crawler.extractor import extract_links

    html = '<a href="https://other.com/page">External</a>'
    links = extract_links(html, "https://example.com", same_domain=False)
    assert "https://other.com/page" in links


def test_extract_links_deduplicates():
    from dewie.crawler.extractor import extract_links

    html = '<a href="/page">1</a><a href="/page">2</a>'
    links = extract_links(html, "https://example.com")
    assert links.count("https://example.com/page") == 1


def test_extract_links_strips_fragment():
    from dewie.crawler.extractor import extract_links

    html = '<a href="/page#section">link</a>'
    links = extract_links(html, "https://example.com")
    assert all("#" not in l for l in links)


def test_extract_links_skips_non_http():
    from dewie.crawler.extractor import extract_links

    html = '<a href="mailto:test@example.com">email</a><a href="ftp://example.com">ftp</a>'
    links = extract_links(html, "https://example.com", same_domain=False)
    assert len(links) == 0


def test_extract_links_empty_html():
    from dewie.crawler.extractor import extract_links

    links = extract_links("", "https://example.com")
    assert links == []


def test_extract_links_no_anchors():
    from dewie.crawler.extractor import extract_links

    html = "<p>No links here</p>"
    links = extract_links(html, "https://example.com")
    assert links == []


def test_extract_links_multiple_same_domain():
    from dewie.crawler.extractor import extract_links

    html = '<a href="/a">a</a><a href="/b">b</a><a href="/c">c</a>'
    links = extract_links(html, "https://example.com")
    assert len(links) == 3
