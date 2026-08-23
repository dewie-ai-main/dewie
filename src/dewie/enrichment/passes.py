# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Concrete enrichment pass implementations.

This module provides the three built-in enrichment passes:

- ``MetadataPass`` — LLM extraction of title, summary, keywords, entities, etc.
- ``ChunkPass``    — Split long docs into chunks with per-chunk embeddings.
- ``EmbedPass``    — Embed the document body for vector search.

Each pass is a standalone ``EnrichmentPass`` that can be registered
in a ``PassRegistry`` and composed into a pipeline.

Usage
-----
::

    from dewie.enrichment.passes import MetadataPass, ChunkPass, EmbedPass
    from dewie.enrichment.registry import PassRegistry

    registry = PassRegistry()
    registry.register(MetadataPass(router, backend_registry))
    registry.register(EmbedPass())
    registry.register(ChunkPass())

    # Or load from config:
    # PassRegistry.from_config(settings, router, backend_registry)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING

from dewie.config import settings
from dewie.debug import dump_step
from dewie.enrichment.base import EnrichmentPass
from dewie.enrichment.processor import MetadataProcessor
from dewie.pipeline import add_edges_for_doc, build_embed_text, embed_batch

if TYPE_CHECKING:
    from dewie.enrichment.registry import BackendRegistry
    from dewie.enrichment.router import EnrichmentRouter
    from dewie.models.content import ContentDocument
    from dewie.storage.postgres import PostgresClient

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL = "text-embedding-3-small"


class MetadataPass(EnrichmentPass):
    """
    LLM-based metadata extraction pass.

    Calls the configured backend pipeline to extract title, summary,
    keywords, entities, sentiment, tone, reading level, and other
    enrichment fields from the document body.

    This is the primary enrichment pass — all other passes assume
    these metadata fields are populated.

    Args:
        router:           ``EnrichmentRouter`` that selects a backend per document.
        registry:         ``BackendRegistry`` used for backend lookup.
        fallback_backend: Legacy parameter, kept for API compat (ignored).
        max_retries:      Maximum backend attempts before marking FAILED.
    """

    name = "metadata"

    def __init__(
        self,
        router: EnrichmentRouter,
        registry: BackendRegistry,
        fallback_backend: str | None = None,
        max_retries: int = 2,
    ) -> None:
        self._processor = MetadataProcessor(
            router=router,
            registry=registry,
            fallback_backend_name=fallback_backend,
            max_retries=max_retries,
        )

    async def run(self, doc: ContentDocument, pg: PostgresClient) -> None:  # type: ignore[override]
        """
        Run LLM extraction on the document.

        On success, populates all metadata enrichment fields and sets
        the document status to ``READY``. On failure, sets status to
        ``FAILED``.
        """
        await self._processor.enrich(doc, pg=pg)


class EmbedPass(EnrichmentPass):
    """
    Document embedding pass.

    Builds the embedding text from the document's metadata and body,
    calls the embedding API, and stores the resulting vector in PostgreSQL.

    Also computes and persists Jaccard-similarity edges to
    ``document_edges`` via the inverted-index SQL path.
    """

    name = "embed"

    async def run(self, doc: ContentDocument, pg: PostgresClient) -> None:  # type: ignore[override]
        """
        Embed the enriched document and compute relationships.

        Steps:
        1. Build embedding text from title, summary, answers_questions, body.
        2. Call the embedding provider.
        3. Store the vector via ``pg.set_embedding()``.
        4. Compute Jaccard-similarity edges via ``add_edges_for_doc()``.
        """
        import httpx

        doc.enriched_at = datetime.utcnow()
        doc.embedding_model = settings.embed_model or _EMBEDDING_MODEL

        # Persist the enriched document to DB
        await pg.upsert(doc)

        # Ensure aq_tsvec is always in sync with answers_questions.
        if doc.answers_questions:
            try:
                from sqlalchemy import text as _text

                if not getattr(pg, "_is_sqlite", False):
                    async with pg._engine.begin() as _conn:
                        await _conn.execute(
                            _text("""
                            UPDATE documents SET aq_tsvec = (
                                SELECT to_tsvector('english', string_agg(v, ' '))
                                FROM jsonb_array_elements_text(
                                    CASE WHEN jsonb_typeof(answers_questions) = 'array'
                                    THEN answers_questions ELSE '[]'::jsonb END
                                ) AS v
                            )
                            WHERE id = cast(:doc_id as uuid)
                        """),
                            {"doc_id": str(doc.id)},
                        )
            except Exception as _tsvec_exc:
                logger.warning("aq_tsvec refresh failed for %s: %s", doc.id, _tsvec_exc)

        # Build embedding text
        embed_text = build_embed_text(
            doc.title,
            doc.summary,
            doc.answers_questions or [],
            doc.body or "",
            doc.embed_summary,
        )

        full_vectors: list[list[float]] = []
        async with httpx.AsyncClient() as http_client:
            vectors = await embed_batch(
                http_client,
                [embed_text],
                full_out=full_vectors if settings.embed_store_full_vector else None,
            )

        if vectors:
            await pg.set_embedding(doc.id, vectors[0])
            if full_vectors:
                await pg.set_embedding_full(doc.id, full_vectors[0])
            # Record "model:dims" after the actual vector is known.
            # This captures MRL truncation (OpenAI/Qwen with EMBED_DIMENSIONS)
            # and fixed-dim models (local/ollama/custom) accurately.
            actual_label = f"{settings.embed_model or _EMBEDDING_MODEL}:{len(vectors[0])}"
            doc.embedding_model = actual_label
            try:
                from sqlalchemy import text as _text
                _id_clause = (
                    "WHERE id = :id"
                    if getattr(pg, "_is_sqlite", False)
                    else "WHERE id = cast(:id as uuid)"
                )
                async with pg._session_factory() as _s:
                    await _s.execute(
                        _text(f"UPDATE documents SET embedding_model = :m {_id_clause}"),
                        {"m": actual_label, "id": str(doc.id)},
                    )
                    await _s.commit()
            except Exception as _em_exc:
                logger.warning("embedding_model label update failed: %s", _em_exc)
            dump_step(
                doc.id,
                "05_embedding",
                {
                    "doc_id": str(doc.id),
                    "vector_length": len(vectors[0]),
                    "model": actual_label,
                },
            )
        else:
            dump_step(
                doc.id,
                "05_embedding",
                {
                    "doc_id": str(doc.id),
                    "vector_length": 0,
                    "model": settings.embed_model or _EMBEDDING_MODEL,
                    "skipped": True,
                },
            )

        # Per-AQ embeddings for the answers_questions_rrf ranker
        # (document_aq table; one vector per AQ string).
        aqs = [q for q in (doc.answers_questions or []) if q and q.strip()]
        if aqs and not getattr(pg, "_is_sqlite", False):
            try:
                async with httpx.AsyncClient() as http_client:
                    aq_vectors = await embed_batch(http_client, aqs)
                if aq_vectors and len(aq_vectors) == len(aqs):
                    await pg.upsert_aq_embeddings(str(doc.id), list(zip(aqs, aq_vectors)))
            except Exception as _aq_exc:
                logger.warning("per-AQ embedding failed for %s: %s", doc.id, _aq_exc)

        # Compute and persist edges
        edge_count = await add_edges_for_doc(pg._engine, str(doc.id))
        dump_step(doc.id, "06_relationships", {"edge_count": edge_count})


class ChunkPass(EnrichmentPass):
    """
    Document chunking pass.

    For long documents (>3k words), splits the body into overlapping
    chunks and stores per-chunk embeddings.  Short documents are
    handled as a single-chunk case.

    Uses the ``chunk_embedder`` module for the actual chunking and
    embedding logic.
    """

    name = "chunk"

    async def run(self, doc: ContentDocument, pg: PostgresClient) -> None:  # type: ignore[override]
        """
        Chunk the document body and store per-chunk embeddings.

        Only processes documents with body text that haven't already
        been chunked.  Documents with empty bodies are marked skipped.
        """
        from dewie.enrichment.chunk_embedder import embed_and_store_chunks

        if not doc.body:
            logger.debug("ChunkPass: no body for doc %s — skipping", doc.id)
            await pg.mark_chunk_status(doc.id, "skipped")
            return

        try:
            ok = await embed_and_store_chunks(
                pg,
                str(doc.id),
                doc.title,
                doc.source,
                doc.body,
            )
            if not ok:
                logger.warning("ChunkPass: chunking failed for doc %s", doc.id)
        except Exception as exc:
            logger.error("ChunkPass: unexpected error for doc %s: %s", doc.id, exc)
            await pg.mark_chunk_status(doc.id, "failed")


def _save_raw_document(doc: ContentDocument) -> None:
    """
    Persist raw document body to disk when ``save_raw_documents`` is enabled.

    Writes to ``./ingested_docs/<source>/<doc_id>.txt``.
    Directory is auto-created. Silently ignores errors.
    """
    if not settings.save_raw_documents:
        return

    out_dir = os.path.join("ingested_docs", str(doc.source))
    os.makedirs(out_dir, exist_ok=True)
    try:
        with open(os.path.join(out_dir, f"{doc.id}.txt"), "w", encoding="utf-8") as f:
            f.write(doc.body or "")
    except Exception:
        logger.exception("Failed to save raw document %s", doc.id)
