"""
Unit tests for src/dewie/ingest/podcast.py

All tests run without network access, real audio files, or a live database.
feedparser, whisper, and psycopg2 are mocked throughout.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_entry(
    title="Test Episode",
    audio_url="https://cdn.example.com/ep1.mp3",
    guid="guid-ep-001",
    pub_date_parsed=None,
    summary="A great episode",
    itunes_duration="45:00",
    itunes_episode=1,
    itunes_season=1,
):
    """Build a fake feedparser entry dict."""
    entry = {
        "title": title,
        "id": guid,
        "summary": summary,
        "enclosures": [{"url": audio_url, "type": "audio/mpeg"}],
        "itunes_duration": itunes_duration,
        "itunes_episode": itunes_episode,
        "itunes_season": itunes_season,
    }
    if pub_date_parsed is not None:
        import time

        t = pub_date_parsed.timetuple()
        entry["published_parsed"] = time.struct_time(t)
    return entry


def _make_parsed_feed(entries):
    """Build a minimal feedparser parse result."""
    return SimpleNamespace(
        entries=entries,
        bozo=False,
        bozo_exception=None,
    )


# ── 1. Feed parsing ────────────────────────────────────────────────────────────


class TestFetchFeed:
    def test_extracts_episode_fields(self):
        """fetch_feed returns correct fields for a standard RSS entry."""
        from dewie.ingestion.podcast import fetch_feed

        pub = datetime(2024, 3, 15, 10, 0, tzinfo=UTC)
        entry = _make_entry(
            title="Deep Dive #42",
            audio_url="https://cdn.example.com/ep42.mp3",
            guid="ep-42-guid",
            pub_date_parsed=pub,
            itunes_duration="1:02:30",
            itunes_episode=42,
            itunes_season=2,
        )
        fake_feed = _make_parsed_feed([entry])

        with patch("feedparser.parse", return_value=fake_feed):
            episodes = fetch_feed("https://feeds.example.com/podcast.rss")

        assert len(episodes) == 1
        ep = episodes[0]
        assert ep["title"] == "Deep Dive #42"
        assert ep["audio_url"] == "https://cdn.example.com/ep42.mp3"
        assert ep["guid"] == "ep-42-guid"
        assert ep["episode_number"] == 42
        assert ep["season"] == 2
        assert ep["duration_seconds"] == 3750  # 1h 2m 30s

    def test_sorts_newest_first(self):
        """Episodes are returned sorted by pub_date descending."""
        from dewie.ingestion.podcast import fetch_feed

        older = _make_entry(
            title="Old", guid="old", pub_date_parsed=datetime(2023, 1, 1, tzinfo=UTC)
        )
        newer = _make_entry(
            title="New", guid="new", pub_date_parsed=datetime(2024, 6, 1, tzinfo=UTC)
        )
        fake_feed = _make_parsed_feed([older, newer])

        with patch("feedparser.parse", return_value=fake_feed):
            episodes = fetch_feed("https://feeds.example.com/podcast.rss")

        assert episodes[0]["title"] == "New"
        assert episodes[1]["title"] == "Old"

    def test_skips_entries_without_audio(self):
        """Entries with no audio enclosure are excluded."""
        from dewie.ingestion.podcast import fetch_feed

        no_audio = {
            "title": "Blog post",
            "id": "blog-1",
            "enclosures": [],
            "links": [],
            "summary": "text",
        }
        has_audio = _make_entry(title="Audio ep", guid="audio-1")
        fake_feed = _make_parsed_feed([no_audio, has_audio])

        with patch("feedparser.parse", return_value=fake_feed):
            episodes = fetch_feed("https://feeds.example.com/podcast.rss")

        assert len(episodes) == 1
        assert episodes[0]["title"] == "Audio ep"

    def test_atom_enclosure_link(self):
        """Atom feeds using link[rel=enclosure] are supported."""
        from dewie.ingestion.podcast import fetch_feed

        entry = {
            "title": "Atom Episode",
            "id": "atom-ep-1",
            "summary": "desc",
            "enclosures": [],
            "links": [
                {"rel": "alternate", "href": "https://example.com/post/1"},
                {"rel": "enclosure", "href": "https://cdn.example.com/atom-ep.mp3"},
            ],
        }
        fake_feed = _make_parsed_feed([entry])

        with patch("feedparser.parse", return_value=fake_feed):
            episodes = fetch_feed("https://feeds.example.com/atom.xml")

        assert len(episodes) == 1
        assert episodes[0]["audio_url"] == "https://cdn.example.com/atom-ep.mp3"


# ── 2. Deduplication ──────────────────────────────────────────────────────────


class TestDeduplication:
    @pytest.mark.asyncio
    async def test_existing_url_is_skipped(self):
        """An episode whose doc_url already exists in the DB is skipped, not re-transcribed."""
        from dewie.ingestion.podcast import _build_doc_url, ingest_podcast_feed

        feed_url = "https://feeds.example.com/podcast.rss"
        pub = datetime(2024, 1, 1, tzinfo=UTC)
        entry = _make_entry(guid="existing-ep", pub_date_parsed=pub)

        doc_url = _build_doc_url(feed_url, "existing-ep")

        # DB already has this URL
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = [(doc_url,)]

        fake_feed = _make_parsed_feed([entry])

        with patch("feedparser.parse", return_value=fake_feed):
            results = await ingest_podcast_feed(feed_url, mock_conn, limit=10)

        assert len(results) == 1
        assert results[0]["status"] == "skipped"
        # No INSERT should have been called
        assert mock_cursor.execute.call_count == 1  # only the SELECT


# ── 3. corpus_id derivation ───────────────────────────────────────────────────


class TestCorpusIdDerivation:
    def test_default_corpus_id_from_feed_url(self):
        """corpus_id defaults to podcast:<host> derived from the feed URL."""
        from dewie.ingestion.podcast import _sanitize_host

        assert _sanitize_host("https://feeds.simplecast.com/abc123") == "feeds.simplecast.com"
        assert _sanitize_host("https://www.example.com/feed.rss") == "example.com"
        assert _sanitize_host("https://anchor.fm/s/abc/podcast/rss") == "anchor.fm"

    @pytest.mark.asyncio
    async def test_custom_corpus_id_is_used(self):
        """Passing corpus_id overrides the default."""
        from dewie.ingestion.podcast import ingest_podcast_feed

        feed_url = "https://feeds.example.com/podcast.rss"
        entry = _make_entry(guid="ep-1")
        fake_feed = _make_parsed_feed([entry])

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = []

        with (
            patch("feedparser.parse", return_value=fake_feed),
            patch("dewie.ingestion.podcast.transcribe_episode", return_value="transcript " * 60),
        ):
            await ingest_podcast_feed(feed_url, mock_conn, limit=1, corpus_id="my-custom-corpus")

        # The SELECT should have been called with our custom corpus_id
        # call_args_list[0][0] is (sql_string, params_tuple)
        select_call_args = mock_cursor.execute.call_args_list[0][0]
        # params_tuple is the second positional arg to cursor.execute
        assert "my-custom-corpus" in select_call_args[1]


# ── 4. URL construction ───────────────────────────────────────────────────────


class TestUrlConstruction:
    def test_url_pattern(self):
        """_build_doc_url returns podcast:<host>:ep:<guid> pattern."""
        from dewie.ingestion.podcast import _build_doc_url

        url = _build_doc_url("https://feeds.example.com/podcast.rss", "ep-123-abc")
        assert url.startswith("podcast:feeds.example.com:ep:")
        assert "ep-123-abc" in url

    def test_url_strips_https_from_guid(self):
        """GUIDs that are HTTP URLs are normalised (protocol stripped)."""
        from dewie.ingestion.podcast import _build_doc_url

        url = _build_doc_url(
            "https://feeds.example.com/podcast.rss",
            "https://example.com/episodes/42",
        )
        assert "https://" not in url
        assert url.startswith("podcast:feeds.example.com:ep:")

    def test_very_long_guid_is_hashed(self):
        """GUIDs exceeding 200 chars are replaced by their SHA-1 hash."""
        from dewie.ingestion.podcast import _build_doc_url

        long_guid = "x" * 300
        url = _build_doc_url("https://feeds.example.com/podcast.rss", long_guid)
        # SHA-1 hex is 40 chars
        parts = url.split(":ep:")
        assert len(parts[1]) == 40


# ── 5. dry_run — no DB writes ─────────────────────────────────────────────────


class TestDryRun:
    @pytest.mark.asyncio
    async def test_dry_run_does_not_write(self):
        """dry_run=True skips all DB writes."""
        from dewie.ingestion.podcast import ingest_podcast_feed

        feed_url = "https://feeds.example.com/podcast.rss"
        entry = _make_entry(guid="ep-dry")
        fake_feed = _make_parsed_feed([entry])

        mock_conn = MagicMock()

        with (
            patch("feedparser.parse", return_value=fake_feed),
            patch("dewie.ingestion.podcast.transcribe_episode", return_value="word " * 100),
        ):
            results = await ingest_podcast_feed(feed_url, mock_conn, limit=1, dry_run=True)

        # conn.cursor() should never have been called
        mock_conn.cursor.assert_not_called()
        assert results[0]["status"] == "ingested"
        assert results[0]["words"] >= 100


# ── 6. --since filter ─────────────────────────────────────────────────────────


class TestSinceFilter:
    @pytest.mark.asyncio
    async def test_since_filters_old_episodes(self):
        """Episodes published before --since are excluded."""
        from dewie.ingestion.podcast import ingest_podcast_feed

        feed_url = "https://feeds.example.com/podcast.rss"

        old_ep = _make_entry(
            title="Old Episode",
            guid="ep-old",
            pub_date_parsed=datetime(2023, 6, 1, tzinfo=UTC),
        )
        new_ep = _make_entry(
            title="New Episode",
            guid="ep-new",
            pub_date_parsed=datetime(2024, 6, 1, tzinfo=UTC),
        )
        fake_feed = _make_parsed_feed([new_ep, old_ep])

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = []

        since = datetime(2024, 1, 1, tzinfo=UTC)

        with (
            patch("feedparser.parse", return_value=fake_feed),
            patch("dewie.ingestion.podcast.transcribe_episode", return_value="word " * 100),
        ):
            results = await ingest_podcast_feed(feed_url, mock_conn, limit=10, since=since)

        titles = [r["title"] for r in results]
        assert "New Episode" in titles
        assert "Old Episode" not in titles

    @pytest.mark.asyncio
    async def test_since_naive_datetime_treated_as_utc(self):
        """A naive since datetime is treated as UTC without raising."""
        from dewie.ingestion.podcast import ingest_podcast_feed

        feed_url = "https://feeds.example.com/podcast.rss"
        entry = _make_entry(
            guid="ep-naive",
            pub_date_parsed=datetime(2024, 6, 1, tzinfo=UTC),
        )
        fake_feed = _make_parsed_feed([entry])

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_cursor.fetchall.return_value = []

        # Naive since — should not raise
        since_naive = datetime(2024, 1, 1)  # no tzinfo

        with (
            patch("feedparser.parse", return_value=fake_feed),
            patch("dewie.ingestion.podcast.transcribe_episode", return_value="word " * 100),
        ):
            results = await ingest_podcast_feed(feed_url, mock_conn, limit=10, since=since_naive)

        assert len(results) == 1
