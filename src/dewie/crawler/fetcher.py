# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Fetcher abstraction for the crawler.

StaticFetcher  — httpx + BeautifulSoup (handles static HTML and lightly dynamic sites).
DynamicFetcher — Playwright stub (raises NotImplementedError until Playwright is added).
FetcherRouter  — Tries StaticFetcher first; escalates JS-rendered pages to needs_js hook.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup

from dewie.config import settings
from dewie.models.content import ContentDocument

logger = logging.getLogger(__name__)

# Signals that suggest the page is a JavaScript single-page application.
_JS_SIGNALS = (
    'id="__next"',
    'id="app"',
    'id="root"',
    "data-reactroot",
    "ng-version=",
    "__vue",
)
_MIN_BODY_CHARS = 200


class BaseCrawlFetcher(ABC):
    """Abstract base for all fetcher strategies."""

    @abstractmethod
    async def fetch(self, url: str) -> tuple[ContentDocument, str]:
        """
        Fetch *url* and return ``(ContentDocument, raw_html)``.

        The raw HTML is needed by the coordinator so it can extract links
        without re-fetching the page.
        """
        ...

    async def __aenter__(self) -> BaseCrawlFetcher:
        return self

    async def __aexit__(self, *args: Any) -> None:
        pass


class StaticFetcher(BaseCrawlFetcher):
    """
    httpx-based fetcher suitable for static HTML and lightly dynamic pages
    (Wikipedia, Hacker News, most blogs).
    """

    def __init__(self) -> None:
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> StaticFetcher:
        self._client = httpx.AsyncClient(
            timeout=settings.crawler_request_timeout,
            follow_redirects=True,
            headers={"User-Agent": settings.user_agent},
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def fetch(self, url: str) -> tuple[ContentDocument, str]:
        if self._client is None:
            raise RuntimeError("StaticFetcher must be used as an async context manager.")

        response = await self._client.get(url)
        response.raise_for_status()
        html = response.text

        soup = BeautifulSoup(html, "lxml")

        title = soup.title.string.strip() if soup.title and soup.title.string else ""

        # Remove boilerplate tags before extracting body text
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()

        body = soup.get_text(separator=" ", strip=True)
        source = urlparse(url).netloc

        doc = ContentDocument(url=url, title=title, body=body, source=source)
        return doc, html


class DynamicFetcher(BaseCrawlFetcher):
    """
    Playwright-based fetcher for JavaScript-rendered pages.

    Stub only — raises NotImplementedError until Playwright is added as
    a project dependency and this class is implemented.
    """

    async def fetch(self, url: str) -> tuple[ContentDocument, str]:
        raise NotImplementedError(
            "DynamicFetcher is not yet implemented. "
            "Install playwright and implement this class to handle JS-rendered pages."
        )


class FetcherRouter(BaseCrawlFetcher):
    """
    Routing fetcher that tries StaticFetcher first and detects JS-rendered pages.

    If the fetched body is shorter than _MIN_BODY_CHARS AND the raw HTML
    contains known JS-framework signals, the page is classified as JS-rendered
    and ``on_js_page`` is called (currently logs a warning and returns the
    sparse document so the caller can mark the job as needs_js).
    """

    def __init__(self) -> None:
        self._static = StaticFetcher()

    async def __aenter__(self) -> FetcherRouter:
        await self._static.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._static.__aexit__(*args)

    async def fetch(self, url: str) -> tuple[ContentDocument, str]:
        doc, html = await self._static.fetch(url)
        if self._looks_like_js(doc.body, html):
            logger.warning("JS-rendered page detected (body=%d chars): %s", len(doc.body), url)
            # Tag the doc so coordinator can check
            doc.body = doc.body  # unchanged — coordinator decides what to do
            return doc, html
        return doc, html

    def is_js_rendered(self, body: str, html: str) -> bool:
        """Public helper so coordinator can call this after fetch."""
        return self._looks_like_js(body, html)

    # Hook — override to plug in a JS queue later
    async def on_js_page(self, url: str, job_id: int) -> None:
        logger.warning("JS-rendered page, marking needs_js: %s (job %d)", url, job_id)

    @staticmethod
    def _looks_like_js(body: str, html: str) -> bool:
        if len(body) >= _MIN_BODY_CHARS:
            return False
        html_lower = html.lower()
        return any(signal.lower() in html_lower for signal in _JS_SIGNALS)
