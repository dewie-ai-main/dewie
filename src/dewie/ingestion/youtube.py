# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
youtube.py — YouTube transcript ingester for Dewie.

Fetches transcripts (no API key required) and metadata for YouTube videos,
then upserts them as ContentDocuments with doc_type='video_transcript'.

Usage:
    from dewie.ingestion.youtube import ingest_video, ingest_channel

    # Single video
    doc = await ingest_video("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    # All videos from a channel (up to limit)
    docs = await ingest_channel("https://www.youtube.com/@lexfridman", limit=50)
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import UTC, datetime
from urllib.parse import parse_qs, urlparse

log = logging.getLogger(__name__)

# Languages to try in order when fetching transcripts
_LANG_PRIORITY = ["en", "en-US", "en-GB"]

# Minimum transcript word count to consider usable
_MIN_WORDS = 100


def _extract_video_id(url: str) -> str | None:
    """Extract YouTube video ID from any standard URL format."""
    parsed = urlparse(url)
    if parsed.hostname in ("youtu.be",):
        return parsed.path.lstrip("/").split("?")[0]
    if parsed.hostname in ("www.youtube.com", "youtube.com", "m.youtube.com"):
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]
        if parsed.path.startswith(("/embed/", "/v/")):
            return parsed.path.split("/")[2]
        if parsed.path.startswith("/shorts/"):
            return parsed.path.split("/")[2]
    # Last-ditch: bare video ID
    if re.match(r"^[A-Za-z0-9_-]{11}$", url):
        return url
    return None


def _fetch_transcript(video_id: str) -> str | None:
    """
    Fetch transcript text for a video. Tries manual captions first,
    then auto-generated. Returns plain text or None if unavailable.
    """
    try:
        from youtube_transcript_api import (  # noqa: PLC0415
            NoTranscriptFound,
            TranscriptsDisabled,
            YouTubeTranscriptApi,
        )
    except ImportError:
        log.warning(
            "youtube_transcript_api not installed — transcript fetch skipped. "
            "Install with: pip install youtube-transcript-api"
        )
        return None

    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        # Try manual captions in preferred languages
        transcript = None
        try:
            transcript = transcript_list.find_manually_created_transcript(_LANG_PRIORITY)
        except Exception:
            pass
        # Fall back to auto-generated
        if transcript is None:
            try:
                transcript = transcript_list.find_generated_transcript(_LANG_PRIORITY)
            except Exception:
                pass
        # Fall back to any available transcript + translate
        if transcript is None:
            available = list(transcript_list)
            if available:
                transcript = available[0]
                try:
                    transcript = transcript.translate("en")
                except Exception:
                    pass

        if transcript is None:
            return None

        chunks = transcript.fetch()
        text = " ".join(chunk["text"] for chunk in chunks)
        # Clean up common transcript artifacts
        text = re.sub(r"\[.*?\]", "", text)  # Remove [Music], [Applause], etc.
        text = re.sub(r"\s+", " ", text).strip()
        return text if len(text.split()) >= _MIN_WORDS else None

    except (NoTranscriptFound, TranscriptsDisabled):
        log.debug("No transcript available for %s", video_id)
        return None
    except Exception as exc:
        log.warning("Transcript fetch failed for %s: %s", video_id, exc)
        return None


def _fetch_video_metadata(video_id: str) -> dict:
    """
    Fetch video metadata using yt-dlp (no network call for transcript — that's
    handled separately). Returns dict with title, channel, published_at, etc.
    """
    import yt_dlp

    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": False,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info is None:
                return {}

            # Parse upload date (YYYYMMDD format from yt-dlp)
            published_at = None
            upload_date = info.get("upload_date")
            if upload_date and len(upload_date) == 8:
                try:
                    published_at = datetime(
                        int(upload_date[:4]),
                        int(upload_date[4:6]),
                        int(upload_date[6:8]),
                        tzinfo=UTC,
                    )
                except ValueError:
                    pass

            return {
                "title": info.get("title", f"YouTube video {video_id}"),
                "channel": info.get("uploader") or info.get("channel", ""),
                "channel_url": info.get("uploader_url") or info.get("channel_url", ""),
                "description": (info.get("description") or ""),
                "duration_seconds": info.get("duration"),
                "view_count": info.get("view_count"),
                "published_at": published_at,
                "thumbnail_url": info.get("thumbnail"),
                "tags": info.get("tags") or [],
            }
    except Exception as exc:
        log.warning("Metadata fetch failed for %s: %s", video_id, exc)
        return {}


async def fetch_video(video_url: str) -> dict | None:
    """
    Fetch transcript + metadata for a single YouTube video.

    Returns a dict ready for ContentDocument construction, or None if the
    video has no usable transcript.

    Dict keys: url, title, body, source, doc_type, published_at, author,
               extra (duration, view_count, thumbnail_url, channel_url)
    """
    video_id = _extract_video_id(video_url)
    if not video_id:
        log.warning("Could not extract video ID from URL: %s", video_url)
        return None

    canonical_url = f"https://www.youtube.com/watch?v={video_id}"

    # Run blocking calls in thread pool
    loop = asyncio.get_event_loop()
    transcript, meta = await asyncio.gather(
        loop.run_in_executor(None, _fetch_transcript, video_id),
        loop.run_in_executor(None, _fetch_video_metadata, video_id),
    )

    description = meta.get("description", "")
    if not transcript:
        # Fall back to video description if it meets the minimum word threshold
        if description and len(description.split()) >= _MIN_WORDS:
            log.info("Using description as body for %s (no transcript available)", canonical_url)
            body = description
        else:
            log.info("Skipping %s — no usable transcript or description", canonical_url)
            return None
    else:
        body = transcript

    title = meta.get("title") or f"YouTube video {video_id}"
    channel = meta.get("channel", "")

    return {
        "url": canonical_url,
        "title": title,
        "body": body,
        "source": "youtube",
        "doc_type": "video_transcript",
        "published_at": meta.get("published_at"),
        "author": channel,
        "extra": {
            "video_id": video_id,
            "channel": channel,
            "channel_url": meta.get("channel_url", ""),
            "duration_seconds": meta.get("duration_seconds"),
            "view_count": meta.get("view_count"),
            "thumbnail_url": meta.get("thumbnail_url", ""),
            "description": meta.get("description", ""),
            "tags": meta.get("tags", []),
        },
    }


async def list_channel_videos(channel_url: str, limit: int = 50) -> list[str]:
    """
    Return up to `limit` video URLs from a YouTube channel or playlist.
    Uses yt-dlp flat extraction (fast — no transcript/metadata fetch).

    Handles yt-dlp's nested structure: channel pages return tab playlists
    (Videos, Shorts, Live), so we recurse one level to collect actual video entries.
    Shorts are excluded (< 60s content, low value for corpus).
    """
    import yt_dlp

    loop = asyncio.get_event_loop()

    def _is_video_id(s: str) -> bool:
        return bool(s and re.match(r"^[A-Za-z0-9_-]{11}$", s))

    def _entry_to_url(entry: dict) -> str | None:
        """Extract a watch URL from a flat entry dict, or None if not a real video."""
        # Direct watch URL
        url = entry.get("url", "")
        if url.startswith("https://www.youtube.com/watch?v="):
            return url
        # Short-form ID field
        vid_id = entry.get("id", "")
        if _is_video_id(vid_id):
            return f"https://www.youtube.com/watch?v={vid_id}"
        return None

    def _collect_urls(entries: list[dict], budget: int) -> list[str]:
        """
        Walk a list of yt-dlp flat entries, recursing into sub-playlists
        (e.g. Videos tab, Live tab) but skipping Shorts tabs.
        Returns up to `budget` unique watch URLs.
        """
        urls: list[str] = []
        seen: set[str] = set()

        for entry in entries:
            if len(urls) >= budget:
                break

            entry_type = entry.get("_type", "url")
            title = (entry.get("title") or "").lower()

            if entry_type == "playlist":
                # Skip Shorts tabs entirely
                if "short" in title:
                    continue
                # Recurse into the sub-playlist entries
                sub_entries = entry.get("entries") or []
                for sub in sub_entries:
                    if len(urls) >= budget:
                        break
                    u = _entry_to_url(sub)
                    if u and u not in seen:
                        seen.add(u)
                        urls.append(u)
            else:
                u = _entry_to_url(entry)
                if u and u not in seen:
                    seen.add(u)
                    urls.append(u)

        return urls

    def _list():
        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "playlistend": limit,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(channel_url, download=False)
            if info is None:
                return []
            entries = info.get("entries") or []
            return _collect_urls(entries, limit)

    try:
        return await loop.run_in_executor(None, _list)
    except Exception as exc:
        log.warning("Channel listing failed for %s: %s", channel_url, exc)
        return []
