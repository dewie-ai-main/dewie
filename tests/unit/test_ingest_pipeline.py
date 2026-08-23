"""
Integration-light unit tests for the ingest pipeline.

No live network calls, no real DB — all external I/O is mocked/patched.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from dewie.ingestion.rss import RSSIngester
from dewie.ingestion.web import WebIngester
from dewie.models.content import ContentDocument, ContentStatus, make_doc_id
from dewie.storage.body_store import load_body, save_body

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_feed(*entries):
    """Build a minimal fake feedparser result."""
    feed = MagicMock()
    feed.entries = list(entries)
    return feed


# ── RSS Ingester ───────────────────────────────────────────────────────────────


async def test_rss_ingester_yields_content_documents():
    entries = [
        {"link": "https://example.com/1", "title": "Entry 1", "summary": "Summary 1"},
        {"link": "https://example.com/2", "title": "Entry 2", "summary": "Summary 2"},
    ]
    fake_feed = _make_feed(*entries)

    with patch("dewie.ingestion.rss.feedparser.parse", return_value=fake_feed):
        ingester = RSSIngester()
        docs = [doc async for doc in ingester.fetch("https://example.com/feed")]

    assert len(docs) == 2
    for doc in docs:
        assert isinstance(doc, ContentDocument)
        assert doc.status == ContentStatus.PENDING
        assert doc.url
        assert doc.title


async def test_rss_ingester_skips_entries_without_link():
    entries = [
        {"link": "https://example.com/1", "title": "Entry 1"},
        {"title": "No link, no id"},  # no link or id key → skipped
        {"link": "https://example.com/3", "title": "Entry 3"},
    ]
    fake_feed = _make_feed(*entries)

    with patch("dewie.ingestion.rss.feedparser.parse", return_value=fake_feed):
        ingester = RSSIngester()
        docs = [doc async for doc in ingester.fetch("https://example.com/feed")]

    assert len(docs) == 2


async def test_poll_rss_feed_ingests_entries():
    """Regression: poll_rss_feed must iterate the async-generator fetch() and
    persist docs. A stray `await` on the async generator used to raise
    TypeError, which the poller's broad except swallowed — so every poll
    silently ingested nothing.
    """
    from dewie.ingestion.rss import poll_rss_feed
    from dewie.models.feed import RSSFeed

    fake_feed = _make_feed(
        {"link": "https://example.com/a", "title": "A", "summary": "body a"},
        {"link": "https://example.com/b", "title": "B", "summary": "body b"},
    )
    pg = MagicMock()
    pg.upsert = AsyncMock()
    pg.write_body_text = AsyncMock()
    pg.mark_feed_polled = AsyncMock()

    feed = RSSFeed(url="https://example.com/feed", name="test")

    with patch("dewie.ingestion.rss.feedparser.parse", return_value=fake_feed), \
         patch("dewie.ingestion.rss._fetch_html", AsyncMock(return_value=None)), \
         patch("dewie.ingestion.rss.save_body"):
        await poll_rss_feed(feed, pg, processor=None)

    # Two entries → two upserts. With the await bug this was zero.
    assert pg.upsert.await_count == 2
    pg.mark_feed_polled.assert_awaited_once()


async def test_rss_ingester_sets_source_from_domain():
    entries = [
        {"link": "https://example.com/feed/article", "title": "Article"},
    ]
    fake_feed = _make_feed(*entries)

    with patch("dewie.ingestion.rss.feedparser.parse", return_value=fake_feed):
        ingester = RSSIngester()
        docs = [doc async for doc in ingester.fetch("https://example.com/feed")]

    assert len(docs) == 1
    assert docs[0].source == "example.com"


# ── Web Ingester ───────────────────────────────────────────────────────────────


async def test_web_ingester_yields_single_document():
    html = "<html><head><title>Test Article</title></head><body><p>Hello world</p></body></html>"
    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.url = "https://example.com/article"
    fake_response.headers = {"content-type": "text/html; charset=utf-8"}
    fake_response.text = html
    fake_response.content = html.encode()

    ingester = WebIngester()
    ingester._client.get = AsyncMock(return_value=fake_response)

    docs = [doc async for doc in ingester.fetch("https://example.com/article")]
    await ingester.close()

    assert len(docs) == 1
    doc = docs[0]
    assert isinstance(doc, ContentDocument)
    assert doc.status == ContentStatus.PENDING
    assert doc.url == "https://example.com/article"


# ── Body Store ────────────────────────────────────────────────────────────────


def test_body_store_written_at_ingest_time(tmp_path, monkeypatch):
    monkeypatch.setattr("dewie.storage.body_store.BODIES_DIR", tmp_path)

    doc = ContentDocument(url="https://example.com/body-test", title="Body Test")
    body_text = "This is the full article body text."

    save_body(doc.id, body_text)
    result = load_body(doc.id)

    assert result == body_text


# ── ContentDocument model ─────────────────────────────────────────────────────


def test_content_document_starts_as_pending():
    doc = ContentDocument(url="https://example.com/article", title="Test Article")
    assert doc.status == ContentStatus.PENDING


def test_deterministic_id_used_at_ingest():
    url = "https://example.com/article"
    doc = ContentDocument.from_url(url)
    assert doc.id == make_doc_id(url)


def test_ingest_url_normalization_deduplicates():
    id1 = make_doc_id("https://example.com/article?utm_source=twitter")
    id2 = make_doc_id("https://example.com/article")
    assert id1 == id2
