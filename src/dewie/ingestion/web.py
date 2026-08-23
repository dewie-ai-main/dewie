# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Single-page web ingester using httpx + trafilatura + BeautifulSoup.

Extraction priority:
1. trafilatura  — best full-article extraction for news/blog URLs
2. BeautifulSoup <article>/<p> fallback

Paywall detection runs on every HTML response:
- HTTP 403                            → paywall_type="hard"
- Schema.org isAccessibleForFree:false → paywall_type="metered"
- Known paywall SDK markers            → paywall_type="metered"
- Subscribe-wall text patterns         → paywall_type="soft"
- Body < 200 chars after extraction    → paywall_type="hard"

Binary formats (PDF, DOCX, XLSX, PPTX) bypass paywall detection.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from collections.abc import AsyncIterator
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from dewie.config import settings
from dewie.ingestion.base import BaseIngester
from dewie.ingestion.extractors import get_extractor
from dewie.ingestion.source_filter import is_blocked_source
from dewie.models.content import ContentDocument, ContentStatus

log = logging.getLogger(__name__)


class _SSRFSafeTransport(httpx.AsyncHTTPTransport):
    """
    Validates the resolved IP address at connection time — after DNS resolution,
    before the TCP handshake. This closes the DNS-rebinding window that a
    pre-request hostname check leaves open, and re-runs the check on every
    redirect hop automatically (because redirects re-enter handle_async_request).
    """

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
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

        return await super().handle_async_request(request)


_HEADERS = {"User-Agent": settings.user_agent}
_TIMEOUT = httpx.Timeout(30.0, connect=5.0)  # longer for large PDFs

# Minimum body chars before we call something a hard paywall
_MIN_BODY_CHARS = 200

_PAYWALL_SDK_MARKERS = (
    # "tinypass" alone is NOT sufficient — many sites (e.g. BBC) load tinypass.min.js
    # for analytics without gating content. Require tp.push() which is the actual
    # paywall initialisation call used by Piano/tinypass.
    "tp.push",
    "pbc_article_access",
    "paywall-widget",
    "laterpay",
    "leaky-paywall",
    "fewcents",
    "cleeng",
    "zuora",
    "recurly-js",
    # piano-id is the login widget, not the paywall gate — removed.
)

_PAYWALL_TEXT_PATTERNS = (
    "subscribe to continue reading",
    "sign up to read more",
    "subscribe to read",
    "become a subscriber",
    "this article is for subscribers",
    "already a subscriber? log in",
    "to read the full story",
    "unlock this article",
    # "subscribers only" removed — too broad; appears in nav/sidebar labels on
    # open-access sites (e.g. Ars Technica subscription upsell in nav).
    "subscribe for full access",
    # "premium content" removed — used loosely on many open sites.
    # "members only" removed — same issue.
    "register to read",
)


def _detect_paywall(html_lower: str) -> tuple[bool, str]:
    """Return (paywall_detected, paywall_type) based on HTML signals.

    html_lower must already be .lower()'d to avoid repeated work.
    """
    # Schema.org: isAccessibleForFree = False
    if (
        '"isaccessibleforfree": "false"' in html_lower
        or '"isaccessibleforfree":false' in html_lower
        or '"isaccessibleforfree": false' in html_lower
    ):
        return True, "metered"

    # Known paywall SDK markers
    for marker in _PAYWALL_SDK_MARKERS:
        if marker in html_lower:
            return True, "metered"

    # Soft paywall text
    for pattern in _PAYWALL_TEXT_PATTERNS:
        if pattern in html_lower:
            return True, "soft"

    return False, "none"


def _extract_with_trafilatura(html: str, url: str) -> str:
    """Extract full article body via trafilatura. Returns '' on failure."""
    try:
        import trafilatura  # lazy import — installed but not always needed at module load

        result = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            favor_precision=False,
            favor_recall=True,
            no_fallback=False,
        )
        return result or ""
    except Exception as exc:
        log.debug("trafilatura extraction failed for %s: %s", url, exc)
        return ""


class WebIngester(BaseIngester):
    """Fetch and extract text content from a single web page."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            transport=_SSRFSafeTransport(),
            headers=_HEADERS,
            timeout=_TIMEOUT,
            follow_redirects=True,
            max_redirects=3,
        )

    async def fetch(self, url: str) -> AsyncIterator[ContentDocument]:
        """
        Fetch the URL and yield a ContentDocument.

        Routes to the appropriate extractor based on Content-Type / file extension:
        - PDF  → pdfplumber text extraction
        - DOCX → python-docx paragraph extraction
        - XLSX → openpyxl cell extraction
        - PPTX → python-pptx slide text extraction
        - HTML → trafilatura full-article extraction, then <p> fallback

        Paywall detection runs on all HTML responses.  A document with
        ``paywall_detected=True`` is yielded so the caller can persist a stub
        but skip enrichment.
        """
        if is_blocked_source(url):
            log.info("WebIngester: blocked source %s — skipping", url)
            return

        try:
            response = await self._client.get(url)
        except ValueError:
            # SSRF block or scheme rejection from _SSRFSafeTransport — re-raise
            # so callers can return a proper error rather than silently yielding nothing.
            raise
        except Exception as exc:
            log.warning("WebIngester: fetch error for %s: %s", url, exc)
            return

        # 403 = access denied / hard paywall — persist stub, skip enrichment
        if response.status_code == 403:
            log.info("WebIngester: 403 for %s — marking paywall_detected=hard", url)
            yield ContentDocument(
                url=url,
                title=url,
                body="",
                source=_extract_domain(url),
                status=ContentStatus.PENDING,
                paywall_detected=True,
                paywall_type="hard",
            )
            return

        if response.status_code == 429:
            log.warning("WebIngester: rate limited (429) for %s — skipping", url)
            return

        if response.status_code >= 400:
            log.warning("WebIngester: HTTP %s for %s — skipping", response.status_code, url)
            return

        resolved_url = str(response.url)
        content_type = response.headers.get("content-type", "")

        # ── Binary document formats ─────────────────────────────────────────
        extractor = get_extractor(content_type, resolved_url)
        if extractor:
            try:
                title, body = extractor(response.content)
                if body.strip():
                    log.info(
                        "WebIngester: extracted %s chars from %s (%s)",
                        len(body),
                        resolved_url,
                        content_type.split(";")[0],
                    )
                    yield ContentDocument(
                        url=resolved_url,
                        title=title or resolved_url,
                        body=body,
                        source=_extract_domain(resolved_url),
                        status=ContentStatus.PENDING,
                    )
                else:
                    log.warning("WebIngester: empty body after extraction for %s", resolved_url)
            except Exception as exc:
                log.warning("WebIngester: extraction failed for %s: %s", resolved_url, exc)
            return

        # ── HTML ────────────────────────────────────────────────────────────
        html = response.text
        html_lower = html.lower()

        soup = BeautifulSoup(html, "lxml")
        title_el = soup.find("title") or soup.find("h1")
        title = title_el.get_text(strip=True) if title_el else resolved_url

        # Paywall detection before body extraction (SDK markers / Schema.org)
        paywall_detected, paywall_type = _detect_paywall(html_lower)

        # Try trafilatura first — far better than raw <p> for news articles
        body = _extract_with_trafilatura(html, resolved_url)

        # Fall back to BeautifulSoup <article>/<p>
        if not body:
            container = soup.find("article") or soup
            paragraphs = container.find_all("p")
            body = " ".join(p.get_text(separator=" ", strip=True) for p in paragraphs)

        # Post-extraction paywall check: very short body on a non-paywalled page
        # almost always means the real content is behind a gate
        if not paywall_detected and len(body) < _MIN_BODY_CHARS:
            log.info(
                "WebIngester: body only %d chars for %s — marking paywall=hard",
                len(body),
                resolved_url,
            )
            paywall_detected = True
            paywall_type = "hard"

        if paywall_detected:
            log.info(
                "WebIngester: paywall_type=%s for %s (body=%d chars)",
                paywall_type,
                resolved_url,
                len(body),
            )

        log.info(
            "WebIngester: extracted %d chars from %s (trafilatura+bs4, paywall=%s)",
            len(body),
            resolved_url,
            paywall_type,
        )

        yield ContentDocument(
            url=resolved_url,
            title=title,
            body=body,
            source=_extract_domain(resolved_url),
            status=ContentStatus.PENDING,
            paywall_detected=paywall_detected,
            paywall_type=paywall_type,
        )

    async def close(self) -> None:
        await self._client.aclose()


def _extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc
    except Exception:
        return url
