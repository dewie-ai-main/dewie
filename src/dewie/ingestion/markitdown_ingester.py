# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.
from collections.abc import AsyncIterator

from markitdown import MarkItDown

from dewie.ingestion.base import BaseIngester
from dewie.models.content import ContentDocument, DocumentType


class MarkItDownIngester(BaseIngester):
    def __init__(self) -> None:
        self._md = MarkItDown()

    async def fetch(self, source: str) -> AsyncIterator[ContentDocument]:
        result = self._md.convert(source)

        title = None
        if hasattr(result, "metadata") and result.metadata:
            title = result.metadata.get("title")

        yield ContentDocument(
            url=source,
            body=result.text_content,
            document_type=self._infer_doc_type(source),
            title=title or source,
        )

    def _infer_doc_type(self, source: str) -> DocumentType:
        source_lower = source.lower()
        if "youtube.com" in source_lower or "youtu.be" in source_lower:
            return DocumentType.VIDEO_TRANSCRIPT
        if source_lower.endswith((".mp3", ".wav", ".m4a")):
            return DocumentType.AUDIO_TRANSCRIPT
        if source_lower.endswith((".pdf", ".docx", ".doc", ".pptx", ".xlsx", ".epub")):
            return DocumentType.DOCUMENT
        return DocumentType.DOCUMENT