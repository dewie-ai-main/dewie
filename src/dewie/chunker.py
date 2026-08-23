# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Document chunker — splits long-form body text into overlapping word windows.

Used by chunk_embedder.py to prepare long documents for per-chunk semantic search.
"""

from __future__ import annotations

# Documents shorter than this are not chunked — single-embedding is sufficient.
MIN_WORDS = 3_000  # threshold for multi-chunk splitting; docs below this get a single chunk

# Target window size and overlap (in words).
CHUNK_WORDS = 1_200
OVERLAP_WORDS = 200

# Step between chunk starts; the overlap is the difference.
_STEP = CHUNK_WORDS - OVERLAP_WORDS  # 1,000 words


def chunk_document(body: str, title: str = "", domain: str = "") -> list[str]:
    """
    Split body into overlapping 1,200-word windows with a context header prepended.

    Short documents (< MIN_WORDS) return a single chunk containing the full body.
    This ensures every document with body text is represented in chunk search.

    Each returned string has the format:
        [Source: {domain} | Title: {title} | Chunk {n}/{total}]\\n\\n{chunk_text}
    """
    words = body.split()
    if not words:
        return []

    # Short docs: single chunk = full body text.
    if len(words) < MIN_WORDS:
        return [f"[Source: {domain} | Title: {title} | Chunk 1/1]\n\n{body}"]

    raw_chunks: list[str] = []
    start = 0
    while start < len(words):
        raw_chunks.append(" ".join(words[start : start + CHUNK_WORDS]))
        start += _STEP

    total = len(raw_chunks)
    return [
        f"[Source: {domain} | Title: {title} | Chunk {i + 1}/{total}]\n\n{chunk}"
        for i, chunk in enumerate(raw_chunks)
    ]
