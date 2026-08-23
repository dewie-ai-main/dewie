# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
RSS/Atom feed ingester.

Uses feedparser (synchronous) wrapped in asyncio.to_thread so the async
interface is preserved without blocking the event loop.

Full-article fetching
---------------------
RSS/Atom feeds deliberately truncate body text to drive traffic to the
publisher's site.  After parsing each entry, this ingester follows the
article URL and extracts full body text via trafilatura — the same
readability-style extraction used by serious crawlers.

Paywall detection
-----------------
During extraction, we check for paywall signals (Schema.org isAccessibleForFree,
metered paywall meta tags, short body after extraction, subscribe-wall keywords).
Detected paywalls are flagged on the document but do NOT block ingest — we enrich
whatever text was extractable and note the paywall type in metadata.

Concurrency
-----------
Article fetches run concurrently (up to FETCH_CONCURRENCY=5 per feed) via
asyncio semaphore to avoid hammering a single origin.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from urllib.parse import urlparse

import feedparser

from dewie.config import settings
from dewie.enrichment.processor import MetadataProcessor
from dewie.ingestion.base import BaseIngester
from dewie.ingestion.source_filter import is_blocked_source
from dewie.models.content import ContentDocument, ContentStatus
from dewie.models.feed import RSSFeed
from dewie.storage.body_store import save_body
from dewie.storage.postgres import PostgresClient

log = logging.getLogger(__name__)

# Max concurrent article fetches per feed call
FETCH_CONCURRENCY = 5

# Minimum extracted word count before we consider it a hard paywall
HARD_PAYWALL_MIN_WORDS = 150

# Keywords that indicate a soft/metered paywall prompt
_PAYWALL_KEYWORDS = re.compile(
    r"(subscribe to (read|continue|access)|sign in to (read|continue|view)|"
    r"this article is (for|available to) (subscriber|member|premium)|"
    r"create (a free )?account to (read|continue)|"
    r"already a subscriber|unlock this story|"
    r"you've reached your (free article|monthly limit)|"
    # Removed: "premium content" and "exclusive (content|access)" — too broad,
    # appear routinely on open-access sites.
    r"subscribe for full access|register to read)",
    re.IGNORECASE,
)


class RSSIngester(BaseIngester):
    """Ingest all entries from an RSS or Atom feed URL."""

    async def fetch(self, url: str) -> AsyncIterator[ContentDocument]:
        """
        Parse the feed at *url*, follow each article URL to fetch full body text,
        and yield one ContentDocument per entry.
        """
        feed = await asyncio.to_thread(feedparser.parse, url)
        source = _extract_domain(url)

        sem = asyncio.Semaphore(FETCH_CONCURRENCY)

        async def _process_entry(entry) -> ContentDocument | None:
            link = entry.get("link") or entry.get("id")
            if not link:
                return None
            if is_blocked_source(link):
                return None

            feed_body = _extract_feed_body(entry)
            full_body, paywall_detected, paywall_type, html = await _fetch_full_article(
                link, feed_body, sem
            )

            # published_at: RSS feed date → OG meta → JSON-LD → URL pattern
            published_at = (
                _extract_published_from_feed(entry)
                or (html and _extract_published_from_html(html))
                or _extract_published_from_url(link)
            )

            doc = ContentDocument(
                url=link,
                title=entry.get("title", ""),
                body=full_body,
                source=source,
                status=ContentStatus.PENDING,
                published_at=published_at,
            )

            if paywall_detected:
                doc.paywall_detected = True
                doc.paywall_type = paywall_type

            return doc

        tasks = [_process_entry(entry) for entry in feed.entries]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        for doc in results:
            if doc is not None:
                yield doc


# ── Full-article fetching ──────────────────────────────────────────────────────


async def _fetch_full_article(
    url: str, feed_body: str, sem: asyncio.Semaphore
) -> tuple[str, bool, str, str | None]:
    """
    Fetch the article at *url* and extract full body text via trafilatura.

    Returns:
        (body_text, paywall_detected, paywall_type, raw_html)
        paywall_type: "none" | "soft" | "hard" | "metered"
        raw_html: raw HTML string if fetch succeeded, else None
    """
    async with sem:
        try:
            html = await _fetch_html(url)
        except Exception:
            return feed_body, False, "none", None

    if not html:
        return feed_body, False, "none", None

    paywall_type = _detect_paywall_from_html(html)

    try:
        import trafilatura

        extracted = await asyncio.to_thread(
            trafilatura.extract,
            html,
            include_comments=False,
            include_tables=False,
            no_fallback=False,
        )
    except Exception:
        extracted = None

    body = extracted or feed_body

    if paywall_type == "none" and body:
        word_count = len(body.split())
        if word_count < HARD_PAYWALL_MIN_WORDS and len(feed_body.split()) < 50:
            paywall_type = "hard"

    if paywall_type == "none" and body and _PAYWALL_KEYWORDS.search(body):
        paywall_type = "soft"

    paywall_detected = paywall_type != "none"
    return body or feed_body, paywall_detected, paywall_type, html


async def _fetch_html(url: str) -> str:
    """
    Fetch raw HTML from *url*.

    Uses a realistic browser-like User-Agent. Follows redirects. 10s timeout.
    Raises on non-2xx status or network error.
    """
    import httpx

    headers = {
        "User-Agent": (
            f"{settings.user_agent} "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    async with httpx.AsyncClient(
        follow_redirects=True,
        timeout=10.0,
        headers=headers,
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.text


def _detect_paywall_from_html(html: str) -> str:
    """
    Detect paywall signals in raw HTML before extraction.

    Returns: "none" | "soft" | "metered"
    """
    # Schema.org isAccessibleForFree: false
    if re.search(r'"isAccessibleForFree"\s*:\s*"?[Ff]alse"?', html):
        return "metered"

    # Standard paywall meta tag
    if re.search(r'<meta[^>]+name=["\']paywall["\'][^>]*>', html, re.IGNORECASE):
        return "metered"

    # Piano / Arc Publishing / common paywall JS SDK signals.
    # NOTE: tinypass.com / piano.io alone is NOT sufficient — many sites (e.g. BBC)
    # load the tinypass SDK for analytics without gating content. Require tp.push()
    # which is the actual paywall initialisation call.
    if re.search(r"tp\.push\s*\(", html):
        return "metered"
    if re.search(r'(arcpublishing\.com/paywall|"locked"\s*:\s*true)', html):
        return "metered"

    return "none"


# ── Publication date extraction ───────────────────────────────────────────────

# URL date patterns: /2024/03/25/, /2024-03-25-, /20240325
_URL_DATE_PATTERNS = [
    re.compile(r"/(\d{4})/(\d{1,2})/(\d{1,2})/"),
    re.compile(r"/(\d{4})-(\d{2})-(\d{2})[/-]"),
    re.compile(r"/(\d{4})(\d{2})(\d{2})[/-]"),
]

# OG/meta published time patterns
_OG_PUBLISHED = re.compile(
    r'<meta[^>]+(?:property|name)=["\'](?:article:published_time|'
    r'og:article:published_time|publish[_-]?date|date)["\'][^>]*'
    r'content=["\']([^"\']+)["\']',
    re.IGNORECASE,
)
_OG_PUBLISHED_ALT = re.compile(
    r'<meta[^>]+content=["\']([^"\']+)["\'][^>]*'
    r'(?:property|name)=["\'](?:article:published_time|publish[_-]?date)["\']',
    re.IGNORECASE,
)

# JSON-LD datePublished
_JSONLD_DATE = re.compile(r'"datePublished"\s*:\s*"([^"]+)"', re.IGNORECASE)


def _parse_iso_date(s: str) -> datetime | None:
    """Parse an ISO 8601-ish date string to UTC datetime. Returns None on failure."""
    if not s:
        return None
    s = s.strip()
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M%z",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(s[:25], fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt.astimezone(UTC)
        except ValueError:
            continue
    return None


def _extract_published_from_feed(entry: dict) -> datetime | None:  # type: ignore[type-arg]
    """Extract published_at from feedparser entry's date fields."""
    import time as _time

    for field in ("published_parsed", "updated_parsed", "created_parsed"):
        t = entry.get(field)
        if t:
            try:
                ts = _time.mktime(t)
                return datetime.fromtimestamp(ts, tz=UTC)
            except Exception:
                continue
    return None


def _extract_published_from_html(html: str) -> datetime | None:
    """Extract published_at from OG meta tags or JSON-LD in raw HTML."""
    for pattern in (_OG_PUBLISHED, _OG_PUBLISHED_ALT, _JSONLD_DATE):
        m = pattern.search(html)
        if m:
            dt = _parse_iso_date(m.group(1))
            if dt:
                return dt
    return None


def _extract_published_from_url(url: str) -> datetime | None:
    """Extract published_at from URL path date patterns like /2024/03/25/."""
    for pattern in _URL_DATE_PATTERNS:
        m = pattern.search(url)
        if m:
            try:
                year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if 2000 <= year <= 2030 and 1 <= month <= 12 and 1 <= day <= 31:
                    return datetime(year, month, day, tzinfo=UTC)
            except (ValueError, IndexError):
                continue
    return None


# ── Feed body extraction (fallback) ───────────────────────────────────────────


def _extract_feed_body(entry: dict) -> str:  # type: ignore[type-arg]
    """
    Extract the best available plain-text body from a feed entry.
    Used as fallback when full-article fetch fails.

    Preference order: content[0] → summary → title.
    """
    if content := entry.get("content"):
        raw = content[0].get("value", "")
    elif summary := entry.get("summary"):
        raw = summary
    else:
        raw = entry.get("title", "")

    return re.sub(r"<[^>]+>", " ", raw).strip()


# ── Helpers ────────────────────────────────────────────────────────────────────


def _extract_domain(url: str) -> str:
    """Return the hostname from a URL, e.g. 'feeds.example.com' → 'feeds.example.com'."""
    try:
        return urlparse(url).netloc or url
    except Exception:
        return url


async def poll_rss_feed(feed: RSSFeed, pg: PostgresClient, processor: MetadataProcessor | None) -> None:
    """Fetch and ingest all documents from a feed, then mark it as polled."""
    from dewie.api.routes.ingest import _enrich_batch

    log.info("feed_poll: starting poll for feed %s (%s)", feed.id, feed.url)
    try:
        ingester = RSSIngester()
        docs: list[ContentDocument] = []
        async for doc in ingester.fetch(feed.url):
            if feed.corpus_id and doc.corpus_id is None:
                doc.corpus_id = feed.corpus_id
            if feed.tags:
                doc.tags = list({*doc.tags, *feed.tags})
            docs.append(doc)

        enrichable: list[ContentDocument] = []
        for doc in docs:
            await pg.upsert(doc)
            if not getattr(doc, "paywall_detected", False) and getattr(doc, "body", None):
                save_body(doc.id, doc.body)
                try:
                    await pg.write_body_text(doc.id, doc.body)
                except Exception as exc:
                    log.warning("feed_poll: write_body_text failed for %s: %s", doc.id, exc)
                enrichable.append(doc)

        if enrichable and processor is not None:
            await _enrich_batch(enrichable, pg, processor)

        await pg.mark_feed_polled(feed.id)
        log.info("feed_poll: completed for feed %s — %d docs ingested", feed.id, len(docs))
    except Exception as exc:
        log.exception("feed_poll: failed for feed %s: %s", feed.id, exc)

