# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.
from collections.abc import AsyncIterator

from dewie.ingestion.markitdown_ingester import MarkItDownIngester
from dewie.ingestion.rss import RSSIngester
from dewie.ingestion.web import WebIngester
from dewie.models.content import ContentDocument


class SourceRouter:
    def __init__(self) -> None:
        self._markitdown_ingester = MarkItDownIngester()
        self._web_ingester = WebIngester()
        self._rss_ingester = RSSIngester()

    async def fetch(self, source: str) -> AsyncIterator[ContentDocument]:
        source_lower = source.lower()

        if source_lower.endswith((".pdf", ".docx", ".doc", ".pptx", ".xlsx", ".epub")):
            async for doc in self._markitdown_ingester.fetch(source):
                yield doc
            return

        if source_lower.endswith((".mp3", ".wav", ".m4a")):
            async for doc in self._markitdown_ingester.fetch(source):
                yield doc
            return

        if "youtube.com" in source_lower or "youtu.be" in source_lower:
            async for doc in self._markitdown_ingester.fetch(source):
                yield doc
            return

        if "rss" in source_lower or "feed" in source_lower or ".xml" in source_lower:
            async for doc in self._rss_ingester.fetch(source):
                yield doc
            return

        async for doc in self._web_ingester.fetch(source):
            yield doc

    async def close(self) -> None:
        await self._markitdown_ingester.close()
        await self._web_ingester.close()
        await self._rss_ingester.close()