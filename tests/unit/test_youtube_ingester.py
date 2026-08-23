"""
Unit tests for YouTube transcript ingester (Issue #105).

Tests cover:
- Video ID extraction from various URL formats
- Transcript fetch (mocked)
- Metadata fetch (mocked)
- fetch_video happy path + no-transcript skip
- list_channel_videos (mocked yt-dlp)
- POST /ingest/youtube API endpoint
"""

from __future__ import annotations

import uuid
from datetime import UTC
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from dewie.ingestion.youtube import (
    _extract_video_id,
    fetch_video,
    list_channel_videos,
)

# ── Video ID extraction ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://m.youtube.com/watch?v=dQw4w9WgXcQ&t=123", "dQw4w9WgXcQ"),
        ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),  # bare video ID
        ("https://www.youtube.com/@lexfridman", None),  # channel URL
        ("https://example.com/not-youtube", None),
    ],
)
def test_extract_video_id(url, expected):
    assert _extract_video_id(url) == expected


# ── fetch_video ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_video_returns_none_for_invalid_url():
    result = await fetch_video("https://example.com/not-youtube")
    assert result is None


@pytest.mark.asyncio
async def test_fetch_video_returns_none_when_no_transcript():
    with (
        patch("dewie.ingestion.youtube._fetch_transcript", return_value=None),
        patch("dewie.ingestion.youtube._fetch_video_metadata", return_value={}),
    ):
        result = await fetch_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert result is None


@pytest.mark.asyncio
async def test_fetch_video_happy_path():
    from datetime import datetime

    meta = {
        "title": "Test Video",
        "channel": "Test Channel",
        "channel_url": "https://www.youtube.com/@test",
        "description": "A test video",
        "duration_seconds": 600,
        "view_count": 12345,
        "published_at": datetime(2024, 1, 15, tzinfo=UTC),
        "thumbnail_url": "https://i.ytimg.com/vi/dQw4w9WgXcQ/default.jpg",
        "tags": ["test", "video"],
    }
    transcript = "This is a test transcript with more than one hundred words. " * 5

    with (
        patch("dewie.ingestion.youtube._fetch_transcript", return_value=transcript),
        patch("dewie.ingestion.youtube._fetch_video_metadata", return_value=meta),
    ):
        result = await fetch_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    assert result is not None
    assert result["url"] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert result["title"] == "Test Video"
    assert result["body"] == transcript
    assert result["doc_type"] == "video_transcript"
    assert result["source"] == "youtube"
    assert result["author"] == "Test Channel"
    assert result["published_at"] == meta["published_at"]
    assert result["extra"]["video_id"] == "dQw4w9WgXcQ"
    assert result["extra"]["duration_seconds"] == 600


# ── list_channel_videos ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_channel_videos_returns_urls():
    pytest.importorskip("yt_dlp", reason="requires yt-dlp — optional media backend")
    mock_info = {
        "entries": [
            {"id": "dQw4w9WgXcQ"},
            {"id": "abc12345678"},
            {"id": "xyz98765432"},
        ]
    }

    mock_ydl = MagicMock()
    mock_ydl.__enter__ = MagicMock(return_value=mock_ydl)
    mock_ydl.__exit__ = MagicMock(return_value=False)
    mock_ydl.extract_info = MagicMock(return_value=mock_info)

    with patch("yt_dlp.YoutubeDL", return_value=mock_ydl):
        urls = await list_channel_videos("https://www.youtube.com/@testchannel", limit=10)

    assert len(urls) == 3
    assert "https://www.youtube.com/watch?v=dQw4w9WgXcQ" in urls


@pytest.mark.asyncio
async def test_list_channel_videos_returns_empty_on_error():
    pytest.importorskip("yt_dlp", reason="requires yt-dlp — optional media backend")
    with patch("yt_dlp.YoutubeDL", side_effect=Exception("network error")):
        urls = await list_channel_videos("https://www.youtube.com/@testchannel")
    assert urls == []


# ── POST /ingest/youtube API ───────────────────────────────────────────────────


def _make_youtube_app(pg_mock) -> FastAPI:
    from dewie.api.routes.ingest import router
    from dewie.enrichment.processor import MetadataProcessor

    app = FastAPI()
    app.include_router(router)
    app.state.postgres = pg_mock
    app.state.processor = MagicMock(spec=MetadataProcessor)
    return app


def test_youtube_ingest_single_video_accepted():
    from datetime import datetime

    video_data = {
        "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "title": "Test Video",
        "body": "transcript " * 120,
        "source": "youtube",
        "doc_type": "video_transcript",
        "author": "Test Channel",
        "published_at": datetime(2024, 1, 15, tzinfo=UTC),
        "extra": {"video_id": "dQw4w9WgXcQ"},
    }

    pg = MagicMock()
    pg.get_by_url = AsyncMock(return_value=None)  # not a duplicate
    pg.upsert = AsyncMock()
    pg.write_body_text = AsyncMock()

    client = TestClient(_make_youtube_app(pg))

    with (
        patch("dewie.ingestion.youtube.fetch_video", AsyncMock(return_value=video_data)),
        patch("dewie.ingestion.youtube.list_channel_videos", AsyncMock(return_value=[])),
    ):
        resp = client.post(
            "/ingest/youtube", json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}
        )

    assert resp.status_code == 202
    body = resp.json()
    assert body["accepted"] == 1
    assert body["skipped"] == 0
    assert len(body["doc_ids"]) == 1


def test_youtube_ingest_skips_no_transcript():
    pg = MagicMock()
    pg.get_by_url = AsyncMock(return_value=None)
    pg.upsert = AsyncMock()

    client = TestClient(_make_youtube_app(pg))

    with patch("dewie.ingestion.youtube.fetch_video", AsyncMock(return_value=None)):
        resp = client.post(
            "/ingest/youtube", json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}
        )

    assert resp.status_code == 202
    body = resp.json()
    assert body["accepted"] == 0
    assert body["skipped"] == 1


def test_youtube_ingest_skips_duplicate():
    from dewie.models.content import ContentDocument, ContentStatus

    existing = ContentDocument(
        id=uuid.uuid4(),
        url="https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        title="Already here",
        source="youtube",
        status=ContentStatus.READY,
    )
    video_data = {
        "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "title": "Test Video",
        "body": "transcript " * 120,
        "source": "youtube",
        "doc_type": "video_transcript",
        "author": "Test Channel",
        "published_at": None,
        "extra": {},
    }

    pg = MagicMock()
    pg.get_by_url = AsyncMock(return_value=existing)  # already exists
    pg.upsert = AsyncMock()

    client = TestClient(_make_youtube_app(pg))

    with patch("dewie.ingestion.youtube.fetch_video", AsyncMock(return_value=video_data)):
        resp = client.post(
            "/ingest/youtube", json={"url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}
        )

    assert resp.status_code == 202
    body = resp.json()
    assert body["accepted"] == 0
    assert body["skipped"] == 1
    pg.upsert.assert_not_called()
