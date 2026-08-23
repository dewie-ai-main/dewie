# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
podcast.py — Podcast RSS ingestion with local Whisper transcription.

Downloads episodes from an RSS/Atom feed, transcribes audio using
openai-whisper (or faster-whisper as fallback), and upserts transcripts
as ContentDocuments with status=pending so the enrichment worker picks them up.

Usage:
    from dewie.ingestion.podcast import ingest_podcast_feed
    results = await ingest_podcast_feed(feed_url, pg, limit=10)

Whisper is NOT a hard dependency — install it separately:
    pip install openai-whisper      # primary
    pip install faster-whisper      # fallback
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
import socket
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlparse

log = logging.getLogger(__name__)

# Minimum transcript word count to be considered usable
_MIN_WORDS = 50

# ── Feed parsing ───────────────────────────────────────────────────────────────


def _parse_duration_seconds(raw: str) -> int | None:
    """Parse iTunes/RSS duration string (HH:MM:SS or MM:SS or seconds) → int seconds."""
    if not raw:
        return None
    raw = raw.strip()
    # Plain seconds
    if re.match(r"^\d+$", raw):
        return int(raw)
    # MM:SS or HH:MM:SS
    parts = raw.split(":")
    try:
        parts = [int(p) for p in parts]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
    except ValueError:
        pass
    return None


def _extract_audio_url(entry: dict) -> str | None:  # type: ignore[type-arg]
    """
    Extract the audio enclosure URL from a feedparser entry.

    Checks both RSS 2.0 enclosures and Atom link[rel=enclosure].
    Returns the first audio/* or known podcast file extension.
    """
    # RSS 2.0 enclosures
    for enc in entry.get("enclosures", []):
        url = enc.get("url", "")
        mime = enc.get("type", "")
        if mime.startswith("audio/") or re.search(
            r"\.(mp3|m4a|ogg|opus|wav|aac|flac)(\?|$)", url, re.IGNORECASE
        ):
            return url

    # Atom link[rel=enclosure]
    for link in entry.get("links", []):
        if link.get("rel") == "enclosure":
            url = link.get("href", "")
            if url:
                return url

    return None


def _entry_to_episode(entry: dict) -> dict | None:  # type: ignore[type-arg]
    """
    Convert a feedparser entry to a normalised episode dict.

    Returns None if the entry has no usable audio enclosure.
    """
    audio_url = _extract_audio_url(entry)
    if not audio_url:
        return None

    # Episode GUID — prefer <guid>, fall back to link, then audio URL
    guid = entry.get("id") or entry.get("link") or audio_url

    # Published date
    published_at: datetime | None = None
    import time as _time

    for field in ("published_parsed", "updated_parsed"):
        t = entry.get(field)
        if t:
            try:
                published_at = datetime.fromtimestamp(_time.mktime(t), tz=UTC)
                break
            except Exception:
                pass

    # Episode / season numbers from iTunes namespace
    episode_number: int | None = None
    season: int | None = None
    raw_ep = entry.get("itunes_episode") or entry.get("episode")
    raw_season = entry.get("itunes_season") or entry.get("season")
    try:
        episode_number = int(raw_ep) if raw_ep else None
    except (ValueError, TypeError):
        pass
    try:
        season = int(raw_season) if raw_season else None
    except (ValueError, TypeError):
        pass

    # Duration
    raw_duration = entry.get("itunes_duration") or entry.get("duration") or ""
    duration_seconds = _parse_duration_seconds(str(raw_duration)) if raw_duration else None

    # Description — prefer summary over title
    description = ""
    if entry.get("summary"):
        description = re.sub(r"<[^>]+>", " ", entry["summary"]).strip()
    elif entry.get("content"):
        description = re.sub(r"<[^>]+>", " ", entry["content"][0].get("value", "")).strip()

    return {
        "title": entry.get("title", "Untitled Episode"),
        "description": description,
        "audio_url": audio_url,
        "pub_date": published_at,
        "duration_seconds": duration_seconds,
        "episode_number": episode_number,
        "season": season,
        "guid": guid,
    }


def fetch_feed(feed_url: str) -> list[dict]:  # type: ignore[type-arg]
    """
    Parse an RSS/Atom podcast feed and return a list of episode dicts,
    sorted newest-first by pub_date.

    Each dict has keys: title, description, audio_url, pub_date,
    duration_seconds, episode_number, season, guid.

    Raises ImportError if feedparser is not installed.
    """
    try:
        import feedparser
    except ImportError as exc:
        raise ImportError(
            "feedparser is required for podcast feed parsing. Install it: pip install feedparser"
        ) from exc

    log.debug("Fetching feed: %s", feed_url)
    parsed = feedparser.parse(feed_url)

    if parsed.bozo and not parsed.entries:
        log.warning("Feed parse warning for %s: %s", feed_url, parsed.bozo_exception)

    episodes: list[dict] = []  # type: ignore[type-arg]
    for entry in parsed.entries:
        ep = _entry_to_episode(entry)
        if ep is not None:
            episodes.append(ep)

    # Sort newest-first; episodes without a date go to the end
    episodes.sort(
        key=lambda e: e["pub_date"] or datetime.min.replace(tzinfo=UTC),
        reverse=True,
    )

    log.info("Feed %s: %d episode(s) with audio", feed_url, len(episodes))
    return episodes


# ── Whisper transcription ─────────────────────────────────────────────────────


def is_podcast_transcription_available() -> bool:
    """Check if Whisper backends are installed."""
    from importlib.util import find_spec

    return find_spec("whisper") is not None or find_spec("faster_whisper") is not None


def _transcribe_with_openai_whisper(audio_path: str, model_size: str) -> str:
    """Transcribe using openai-whisper (local model)."""
    import whisper  # type: ignore[import-untyped]

    model = whisper.load_model(model_size)
    result = model.transcribe(audio_path, fp16=False)
    return result.get("text", "").strip()


def _transcribe_with_faster_whisper(audio_path: str, model_size: str) -> str:
    """Transcribe using faster-whisper (CTranslate2-based)."""
    from faster_whisper import WhisperModel  # type: ignore[import-untyped]

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path)
    return " ".join(seg.text for seg in segments).strip()


class _SSRFSafeSyncTransport:
    """Sync httpx transport that validates resolved IPs before connecting."""

    def __new__(cls):
        import httpx as _httpx

        class _Transport(_httpx.HTTPTransport):
            def handle_request(self, request: _httpx.Request) -> _httpx.Response:
                if request.url.scheme not in ("http", "https"):
                    raise ValueError(f"Blocked scheme {request.url.scheme!r} for {request.url.host!r}")
                host = request.url.host
                port = request.url.port or (443 if request.url.scheme == "https" else 80)
                try:
                    infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
                except socket.gaierror as exc:
                    raise ValueError(f"DNS resolution failed for {host!r}") from exc
                for *_, sockaddr in infos:
                    ip = ipaddress.ip_address(sockaddr[0])
                    if ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_reserved or ip.is_multicast:
                        raise ValueError(f"Blocked internal address {ip} ({host!r})")
                return super().handle_request(request)

        return _Transport()


def transcribe_episode(audio_url: str, model_size: str = "base") -> str:
    """
    Download an audio file from *audio_url* and transcribe it with Whisper.

    Tries openai-whisper first; falls back to faster-whisper. Raises
    ImportError with install instructions if neither is available.

    The downloaded audio file is always cleaned up, even on failure.

    Returns the full transcript as a plain-text string.
    """
    import httpx

    suffix = Path(urlparse(audio_url).path).suffix or ".mp3"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Stream download to temp file (SSRF-safe transport validates resolved IP).
        # A podcast-client User-Agent is required: podcast CDNs behind Cloudflare
        # (Buzzsprout, Libsyn, etc.) 403 the default httpx UA *and* browser UAs
        # hitting an mp3 directly (anti-scraping), but serve podcast apps.
        log.debug("Downloading audio: %s → %s", audio_url, tmp_path)
        headers = {
            "User-Agent": "Dewie Podcast Ingester/1.0 (+https://dewie.ai)",
            "Accept": "*/*",
        }
        with httpx.Client(
            transport=_SSRFSafeSyncTransport(), follow_redirects=True,
            timeout=300.0, headers=headers,
        ) as client:
            with client.stream("GET", audio_url) as resp:
                resp.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=65536):
                        f.write(chunk)

        file_size_mb = Path(tmp_path).stat().st_size / (1024 * 1024)
        log.debug("Downloaded %.1f MB, starting transcription (model=%s)", file_size_mb, model_size)

        # Try openai-whisper first
        try:
            return _transcribe_with_openai_whisper(tmp_path, model_size)
        except ImportError:
            pass

        # Fall back to faster-whisper
        try:
            return _transcribe_with_faster_whisper(tmp_path, model_size)
        except ImportError:
            pass

        raise ImportError(
            "No Whisper backend found. Install one:\n"
            "  pip install openai-whisper       # recommended\n"
            "  pip install faster-whisper       # lighter alternative (CTranslate2)"
        )

    finally:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except Exception:
            pass


# ── Corpus ingestion ──────────────────────────────────────────────────────────


def _sanitize_host(url: str) -> str:
    """Extract and lightly sanitize the netloc from a URL for use in identifiers."""
    try:
        host = urlparse(url).netloc
        # Strip leading www.
        host = re.sub(r"^www\.", "", host)
        return host or "unknown"
    except Exception:
        return "unknown"


def _build_doc_url(feed_url: str, guid: str) -> str:
    """
    Build the canonical Dewie URL for a podcast episode.

    Pattern: podcast:<feed_host>:ep:<sanitized_guid>
    """
    host = _sanitize_host(feed_url)
    # Sanitize guid: strip protocol, replace unsafe chars
    clean_guid = re.sub(r"^https?://", "", guid)
    clean_guid = re.sub(r"[^A-Za-z0-9._~:@!$&'()*+,;=/-]", "_", clean_guid)
    # Truncate if very long
    if len(clean_guid) > 200:
        import hashlib

        clean_guid = hashlib.sha1(guid.encode()).hexdigest()
    return f"podcast:{host}:ep:{clean_guid}"


async def ingest_podcast_feed(
    feed_url: str,
    pg,  # PostgresClient (psycopg2 sync connection or async — caller decides)
    limit: int = 20,
    model_size: str = "base",
    dry_run: bool = False,
    corpus_id: str | None = None,
    since: datetime | None = None,
) -> list[dict]:  # type: ignore[type-arg]
    """
    Fetch a podcast feed, transcribe new episodes, and upsert them to DB.

    Parameters
    ----------
    feed_url:   RSS/Atom feed URL.
    pg:         A psycopg2 connection (sync) used for direct DB writes.
    limit:      Max episodes to process (newest first).
    model_size: Whisper model size (tiny/base/small/medium/large).
    dry_run:    Transcribe but skip DB writes.
    corpus_id:  Optional corpus UUID to file episodes under (None = ungrouped;
                episodes are always tagged by feed host via the `source` field).
    since:      Skip episodes published before this datetime.

    Returns a list of result dicts:
        {title, url, words, status: "ingested"|"skipped"|"failed", error?}
    """
    # corpus_id is a UUID column (FK to corpora) — NOT a free-text tag. Episodes
    # of a feed are grouped by their host in the `source` field instead; corpus_id
    # is left as passed (a real corpus UUID, or None).
    feed_source = _sanitize_host(feed_url)

    loop = asyncio.get_event_loop()

    # Fetch feed in thread pool (feedparser is synchronous)
    episodes = await loop.run_in_executor(None, fetch_feed, feed_url)

    # Apply --since filter
    if since is not None:
        # Ensure since is timezone-aware
        if since.tzinfo is None:
            since = since.replace(tzinfo=UTC)
        episodes = [ep for ep in episodes if ep["pub_date"] is None or ep["pub_date"] >= since]

    # Cap to limit
    episodes = episodes[:limit]

    if not episodes:
        log.info("No episodes to process after filters.")
        return []

    # Load existing URLs from DB to deduplicate
    existing_urls: set[str] = set()
    if not dry_run and pg is not None:

        def _load_existing():
            with pg.cursor() as cur:
                cur.execute(
                    "SELECT url FROM documents WHERE source = %s",
                    (feed_source,),
                )
                return {row[0] for row in cur.fetchall()}

        existing_urls = await loop.run_in_executor(None, _load_existing)
        log.debug("Already ingested: %d doc(s) from source %s", len(existing_urls), feed_source)

    results: list[dict] = []  # type: ignore[type-arg]

    for ep in episodes:
        doc_url = _build_doc_url(feed_url, ep["guid"])
        title = ep["title"]

        # Skip already-ingested episodes
        if doc_url in existing_urls:
            log.debug("Skipping (already ingested): %s", doc_url)
            results.append({"title": title, "url": doc_url, "words": 0, "status": "skipped"})
            continue

        # Transcribe
        t0 = time.monotonic()
        transcript = ""
        fallback_mode = False
        try:
            transcript = await loop.run_in_executor(
                None, transcribe_episode, ep["audio_url"], model_size
            )
        except ImportError as exc:
            log.warning("Whisper unavailable, falling back to metadata only for %s: %s", title, exc)
            transcript = ""
            fallback_mode = True
        except Exception as exc:
            log.error("Transcription failed for %s: %s", title, exc)
            results.append(
                {"title": title, "url": doc_url, "words": 0, "status": "failed", "error": str(exc)}
            )
            continue

        elapsed = time.monotonic() - t0
        word_count = len(transcript.split())

        if not fallback_mode and word_count < _MIN_WORDS:
            log.warning("Transcript too short (%d words) for %s, skipping", word_count, title)
            results.append(
                {
                    "title": title,
                    "url": doc_url,
                    "words": word_count,
                    "status": "failed",
                    "error": "transcript too short",
                }
            )
            continue

        log.info("Transcribed: %s — %d words in %.1fs", title, word_count, elapsed)

        if dry_run:
            results.append(
                {"title": title, "url": doc_url, "words": word_count, "status": "ingested"}
            )
            continue

        # Write to DB
        source = _sanitize_host(feed_url)
        published_at = ep["pub_date"]

        extra = {
            "audio_url": ep["audio_url"],
            "duration_seconds": ep["duration_seconds"],
            "episode_number": ep["episode_number"],
            "season": ep["season"],
            "description": ep["description"],
            "feed_url": feed_url,
        }

        def _insert():
            with pg.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents
                        (id, url, title, body_text, source, status, corpus_id,
                         summary, embed_summary, topics, keywords, entities,
                         sentiment, ingested_at, enrichment_version, published_at)
                    VALUES
                        (gen_random_uuid(), %s, %s, %s, %s, 'pending', %s,
                         '', '', '[]', '[]', '[]',
                         0.0, now(), 0, %s)
                    ON CONFLICT (url) DO NOTHING
                    """,
                    (doc_url, title, transcript, source, corpus_id, published_at),
                )
            pg.commit()

        try:
            await loop.run_in_executor(None, _insert)
            existing_urls.add(doc_url)
            results.append(
                {"title": title, "url": doc_url, "words": word_count, "status": "ingested"}
            )
        except Exception as exc:
            log.error("DB insert failed for %s: %s", doc_url, exc)
            results.append(
                {
                    "title": title,
                    "url": doc_url,
                    "words": word_count,
                    "status": "failed",
                    "error": str(exc),
                }
            )

    return results
