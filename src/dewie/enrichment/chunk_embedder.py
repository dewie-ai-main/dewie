# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
chunk_embedder.py — Embed document chunks for all documents with body text.

For each ready document with chunk_status='none', this module:
  1. Loads body text from the flat-file store (falls back to body_text column).
  2. Splits into chunks via dewie.chunker:
     - Short docs (<3,000 words): 1 chunk = full body text
     - Long docs: overlapping 1,200-word windows
  3. Embeds each chunk via the configured EmbeddingProvider (batches of 20).
  4. Validates vector dimensions, truncating if actual > expected.
  5. Retries via API when actual < expected (short embedding fallback).
  6. Inserts rows into document_chunks.
  7. Sets chunk_status = 'chunked' on the document.

On empty body: marks chunk_status = 'skipped'.
On any failure: marks chunk_status = 'failed' so it can be retried.
"""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import urlparse
from uuid import UUID

from dewie.chunker import chunk_document
from dewie.providers.base import EmbeddingDimensionMismatchError
from dewie.providers.factory import get_embedding_provider
from dewie.storage.body_store import load_body
from dewie.storage.postgres import PostgresClient, _embed_dimensions_for_model

log = logging.getLogger(__name__)

EMBED_BATCH = 20  # max chunks per embedding API call


def _domain(source: str) -> str:
    """Extract hostname from a URL-like source string, or return source as-is."""
    try:
        return urlparse(source).hostname or source
    except Exception:
        return source or ""


async def _validate_and_fix_dimensions(
    vectors: list[list[float]], expected_dims: int
) -> list[list[float]]:
    """
    Validate vector dimensions and fix mismatches.

    - If actual > expected: truncate with warning logged.
    - If actual < expected: raise EmbeddingDimensionMismatchError (triggers API retry).
    - If actual == expected: return as-is.
    """
    if not vectors or expected_dims is None:
        return vectors

    first_dim = len(vectors[0]) if vectors else 0
    if first_dim == expected_dims:
        return vectors

    if first_dim > expected_dims:
        log.warning(
            "chunk_embedder: truncating %d-dim embedding to %d dims for doc chunks",
            first_dim, expected_dims,
        )
        return [v[:expected_dims] for v in vectors]

    raise EmbeddingDimensionMismatchError(
        expected=expected_dims, actual=first_dim, model="unknown"
    )


async def embed_and_store_chunks(
    pg: PostgresClient,
    doc_id: str,
    title: str,
    source: str,
    body: str,
) -> bool:
    """
    Chunk, embed, and store chunks for a single document.
    Returns True on success, False on failure.
    """
    chunks = chunk_document(body, title=title, domain=_domain(source))
    if not chunks:
        # Body was shorter than MIN_WORDS — mark as skipped so we know no chunks exist.
        await pg.mark_chunk_status(UUID(doc_id), "skipped")
        return True

    provider = get_embedding_provider()
    chunk_rows: list[tuple[int, str, list[float]]] = []

    # Extract model name from provider to resolve expected dimensions
    model_name = getattr(provider, "model", getattr(provider, "model_name", "text-embedding-3-small"))

    # Determine expected dimensions for the model; also build the stamp string
    # ``model_name:dimensions`` used when writing chunks into the DB.
    expected_dims = None
    try:
        expected_dims = _embed_dimensions_for_model(model_name)
    except Exception:
        expected_dims = None

    embedding_model: str | None = None
    if model_name is not None and expected_dims is not None:
        embedding_model = f"{model_name}:{expected_dims}"

    for i in range(0, len(chunks), EMBED_BATCH):
        batch_texts = chunks[i : i + EMBED_BATCH]
        try:
            vectors = await _embed_with_retry(provider, batch_texts, expected_dims, doc_id, i)
        except Exception as exc:
            log.warning(
                "chunk_embedder: embed failed for doc %s batch %d: %s",
                doc_id,
                i // EMBED_BATCH,
                exc,
            )
            return False
        if vectors is None:
            return False

        if vectors is None or len(vectors) != len(batch_texts):
            log.warning(
                "chunk_embedder: unexpected embed result for doc %s batch %d",
                doc_id,
                i // EMBED_BATCH,
            )
            return False

        for j, (text, vec) in enumerate(zip(batch_texts, vectors)):
            chunk_rows.append((i + j, text, vec))

    await pg.insert_chunks(UUID(doc_id), chunk_rows, embedding_model)
    await pg.mark_chunk_status(UUID(doc_id), "chunked")
    log.info("chunk_embedder: %s chunks stored for doc %s", len(chunk_rows), doc_id)
    return True


async def _embed_with_retry(
    provider, batch_texts: list[str], expected_dims: int | None, doc_id: str, batch_idx: int
) -> list[list[float]] | None:
    """
    Embed a batch of texts with dimension validation and API retry fallback.

    If the embedding is shorter than expected (actual < expected), raises
    EmbeddingDimensionMismatchError which triggers a retry via the API with
    explicit dimensions parameter for OpenAI-compatible providers.
    """
    try:
        vectors = await provider.embed(batch_texts)
        if vectors is None:
            return None

        vectors = await _validate_and_fix_dimensions(vectors, expected_dims)
        return vectors

    except EmbeddingDimensionMismatchError as exc:
        # Attempt API retry with explicit dimensions for OpenAI-compatible providers
        log.warning(
            "chunk_embedder: embedding dimension mismatch (expected=%d, got=%d), "
            "retrying via API for doc %s batch %d",
            exc.expected, exc.actual, doc_id, batch_idx,
        )
        return await _retry_via_api(provider, batch_texts, expected_dims, doc_id, batch_idx)


async def _retry_via_api(
    provider, batch_texts: list[str], expected_dims: int | None, doc_id: str, batch_idx: int
) -> list[list[float]] | None:
    """
    Retry embedding via API when short embeddings are detected.

    For OpenAI-compatible providers, sends the dimensions parameter explicitly
    to get embeddings of the correct size. Falls back to returning None (failure)
    for non-API providers.
    """
    model_name = getattr(provider, "model", getattr(provider, "model_name", "text-embedding-3-small"))

    # Only retry for OpenAI-compatible providers (have an embed URL)
    embed_url = getattr(provider, "_embed_url", None)
    api_key = getattr(provider, "_api_key", None)
    if not embed_url or not api_key:
        log.error(
            "chunk_embedder: cannot retry via API for provider %r — no embed URL or API key",
            getattr(provider, "name", "unknown"),
        )
        return None

    import httpx

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # Merge extra headers if present
    extra_headers = getattr(provider, "_extra_headers", {})
    if extra_headers:
        headers = {**headers, **extra_headers}

    payload: dict = {"model": model_name, "input": batch_texts}
    if expected_dims is not None:
        payload["dimensions"] = expected_dims

    try:
        async with httpx.AsyncClient() as http:
            r = await http.post(
                embed_url,
                headers=headers,
                json=payload,
                timeout=30,
            )
            r.raise_for_status()
            data = r.json()
            vectors = [
                item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])
            ]
            # Validate dimensions again after retry
            vectors = await _validate_and_fix_dimensions(vectors, expected_dims)
            return vectors
    except Exception as e:
        log.warning(
            "chunk_embedder: API retry failed for doc %s batch %d: %s",
            doc_id, batch_idx, e,
        )
        return None


async def run_once(
    pg: PostgresClient,
    batch_size: int = 50,
    skip_ids: set | None = None,
    fail_counts: dict | None = None,
    max_fails: int = 2,
) -> int:
    """
    Process one batch of un-chunked documents.
    Returns the number of documents successfully chunked.
    skip_ids: set of doc IDs to skip this pass (locally-tracked failures).
    """
    docs = await pg.get_unchunked_docs(limit=batch_size + len(skip_ids or set()))
    if not docs:
        return 0

    succeeded = 0
    for doc_id, title, source, db_body in docs:
        if skip_ids and doc_id in skip_ids:
            continue
        body = await asyncio.to_thread(load_body, doc_id)
        if not body:
            # Flat-file missing — fall back to body_text column (distributed workers).
            body = db_body or None
        if not body:
            log.debug("chunk_embedder: no body for doc %s — skipping", doc_id)
            await pg.mark_chunk_status(UUID(doc_id), "skipped")
            continue

        try:
            ok = await embed_and_store_chunks(pg, doc_id, title or "", source or "", body)
            if ok:
                succeeded += 1
                if fail_counts and doc_id in fail_counts:
                    del fail_counts[doc_id]
            else:
                if fail_counts is not None:
                    fail_counts[doc_id] = fail_counts.get(doc_id, 0) + 1
                    if fail_counts[doc_id] >= max_fails:
                        log.warning(
                            "chunk_embedder: doc %s failed %d times, skipping permanently this run",
                            doc_id,
                            fail_counts[doc_id],
                        )
                    else:
                        await pg.mark_chunk_status(UUID(doc_id), "failed")
                else:
                    await pg.mark_chunk_status(UUID(doc_id), "failed")
        except Exception as exc:
            log.error("chunk_embedder: unexpected error for doc %s: %s", doc_id, exc)
            if fail_counts is not None:
                fail_counts[doc_id] = fail_counts.get(doc_id, 0) + 1
                if fail_counts[doc_id] < max_fails:
                    try:
                        await pg.mark_chunk_status(UUID(doc_id), "failed")
                    except Exception:
                        pass
            else:
                try:
                    await pg.mark_chunk_status(UUID(doc_id), "failed")
                except Exception:
                    pass

    return succeeded
