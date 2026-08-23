# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
MetadataProcessor — orchestrates the full document enrichment cycle.

This module is the single entry point for enriching a ``ContentDocument``.
It coordinates:

1. Backend selection via ``EnrichmentRouter``.
2. Prompt construction via ``build_extraction_prompt``.
3. Backend invocation (LLM, spaCy, agent, stub, …).
4. JSON parsing and ``ExtractionResult`` validation.
5. Populating ``ContentDocument`` fields from the extraction result.
6. Fallback to ``SpacyBackend`` if the primary backend fails or returns
   unparseable output.
7. Persistence to PostgreSQL via ``enrich_and_persist``.

Separation of concerns
-----------------------
- This module has **no FastAPI imports**.  It is safe to call from the crawler
  coordinator, a background task, a queue consumer, or a CLI script.
- HTTP and routing concerns live in their respective modules.
- Storage concerns in ``enrich_and_persist`` are injected (PostgresClient,
  GraphClient) so the processor is unit-testable with mocks.

Retry / fallback behaviour
---------------------------
1. Primary backend is selected by the router.
2. If the primary backend raises ``BackendError`` or returns non-JSON, the
   fallback backend (``settings.enrichment_fallback_backend``, default ``spacy``)
   is attempted.
3. If the fallback also fails, the document is marked ``FAILED`` and a
   structured log entry is emitted.
4. The retry count is capped at ``settings.enrichment_max_retries``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from typing import TYPE_CHECKING

_EMBEDDING_MODEL = "text-embedding-3-small"

from dewie.config import settings
from dewie.debug import dump_step
from dewie.enrichment.base import BackendError, ExtractionResult
from dewie.enrichment.schema import (
    MAX_EMBED_SUMMARY_CHARS,
    MAX_SUMMARY_CHARS,
    build_extraction_prompt,
)
from dewie.models.content import ContentDocument, ContentStatus, DocumentType, ReadingLevel
from dewie.storage.llm_cache import get_cached, set_cached

if TYPE_CHECKING:
    from dewie.enrichment.registry import BackendRegistry
    from dewie.enrichment.router import EnrichmentRouter
    from dewie.storage.postgres import PostgresClient

logger = logging.getLogger(__name__)


class MetadataProcessor:
    """
    Stateful enrichment orchestrator.

    Inject ``router`` and ``registry`` at construction time; the processor
    then uses them for every ``enrich()`` call.  Both dependencies are
    typically built from config at application startup and shared across
    all requests.

    Args:
        router:   ``EnrichmentRouter`` that selects a backend per document.
        registry: ``BackendRegistry`` used for fallback backend lookup.
        max_retries: Maximum number of backend attempts before marking FAILED.

    Note: spaCy fallback has been removed. LLM failures fail hard — no silent
    garbage data from NLP heuristics.
    """

    def __init__(
        self,
        router: EnrichmentRouter,
        registry: BackendRegistry,
        fallback_backend_name: str | None = None,  # kept for signature compat, ignored
        max_retries: int = 2,
    ) -> None:
        self._router = router
        self._registry = registry
        self._fallback_backend_name = None  # spaCy removed — no fallback
        self._max_retries = max_retries

    # ── Public API ────────────────────────────────────────────────────────────

    async def enrich(self, doc: ContentDocument, pg=None) -> ContentDocument:
        """
        Enrich a document using the configured backend pipeline.

        Selects the appropriate backend via the router, builds the extraction
        prompt, calls the backend, parses the result, and populates the
        document's enrichment fields.

        On backend failure or JSON parse error:
        - Logs the error with structured context.
        - Retries using the fallback backend.
        - Marks the document ``FAILED`` if all attempts are exhausted.

        Args:
            doc: Document to enrich.  Must have ``title`` and ``body``
                 populated.  The ``body`` field is read here but not persisted.

        Returns:
            The same ``doc`` with enrichment fields populated and status set
            to ``READY`` (on success) or ``FAILED`` (on total failure).
            Status is set to ``TERMINAL`` if the body is empty/missing —
            these are moved to review_queue and never retried.
        """
        if not doc.body or not doc.body.strip():
            logger.warning("Document %s has empty body; marking terminal.", doc.url)
            doc.status = ContentStatus.TERMINAL
            return doc

        # Language gate — skip enrichment for non-English docs before any LLM call
        try:
            from langdetect import LangDetectException, detect
            try:
                lang = detect(doc.body[:2000])
                if lang != "en":
                    logger.info("Document %s detected as %r — marking terminal.", doc.url, lang)
                    doc.status = ContentStatus.TERMINAL
                    doc.skip_reason = f"non_english:{lang}"
                    return doc
            except LangDetectException:
                pass  # too short or ambiguous — proceed to LLM
        except ImportError:
            pass  # langdetect not installed — gate disabled

        # Build the prompt once; reuse across retry attempts
        prompt = build_extraction_prompt(doc.title, doc.body)

        primary = self._router.select(doc)
        # No fallback — spaCy removed. Fail hard on LLM failure.
        backends_to_try = [primary]

        # Attempt backend(s) in order
        last_error: Exception | None = None
        for attempt, backend in enumerate(backends_to_try[: self._max_retries + 1], 1):
            try:
                if pg is not None:
                    cached = await get_cached(pg, doc.id, "extraction", backend.name, prompt)
                    if cached is not None:
                        logger.info(
                            "LLM cache hit for %s (backend=%s, attempt=%d)",
                            doc.url,
                            backend.name,
                            attempt,
                        )
                        raw = cached
                    else:
                        raw = await backend.complete(prompt)
                        await set_cached(pg, doc.id, "extraction", backend.name, prompt, raw)
                else:
                    raw = await backend.complete(prompt)
                result = _parse_extraction_result(raw)
                dump_step(doc.id, "02_llm_extraction", result.model_dump())
                _apply_result_to_doc(doc, result)
                if settings.enrichment_mode == "dual_pass":
                    await _run_dual_pass(doc)
                    dump_step(
                        doc.id,
                        "02b_dual_pass",
                        {
                            "answers_questions": (doc.answers_questions or [])[:3],
                            "keywords": (doc.keywords or [])[:5],
                            "entities": (doc.entities or [])[:5],
                        },
                    )
                dump_step(
                    doc.id,
                    "03_field_population",
                    {
                        "document_type": doc.document_type.value if doc.document_type else None,
                        "summary": (doc.summary or "")[:120],
                        "topics": doc.topics,
                        "keywords": doc.keywords[:5],
                        "entities": doc.entities[:5],
                        "sentiment": doc.sentiment,
                        "tone": doc.tone,
                        "reading_level": doc.reading_level.value if doc.reading_level else None,
                        "author": doc.author,
                        "language": doc.language,
                    },
                )
                doc.status = ContentStatus.READY
                logger.info(
                    "Enrichment succeeded for %s (backend=%s, attempt=%d)",
                    doc.url,
                    backend.name,
                    attempt,
                )
                return doc

            except (BackendError, json.JSONDecodeError, ValueError) as exc:
                last_error = exc
                logger.warning(
                    "Enrichment attempt %d/%d failed for %s (backend=%s): %s",
                    attempt,
                    len(backends_to_try),
                    doc.url,
                    backend.name,
                    exc,
                )

        # All attempts exhausted
        logger.error(
            "All enrichment attempts failed for %s. Last error: %s",
            doc.url,
            last_error,
        )
        doc.status = ContentStatus.FAILED
        return doc

    async def enrich_and_persist(
        self,
        doc: ContentDocument,
        pg: PostgresClient,
    ) -> None:
        """
        Enrich a document and persist it to PostgreSQL.

        Steps:
        1. Mark document as ``PROCESSING`` in PostgreSQL.
        2. Run ``enrich(doc)`` with the configured backend pipeline.
        3. Upsert the enriched document to PostgreSQL.
        4. Find candidate documents for relationship building.
        5. Compute and persist Jaccard-similarity edges to document_edges.

        Args:
            doc: Document to enrich.  ``body`` must be populated.
            pg:  Async PostgreSQL client.
        """
        import httpx

        from dewie.pipeline import add_edges_for_doc, build_embed_text, embed_batch

        # Step 01 — body load (pre-loaded by caller; snapshot length for debug)
        dump_step(
            doc.id,
            "01_body_load",
            {
                "doc_id": str(doc.id),
                "body_length": len(doc.body or ""),
                "source": "pre-loaded",
            },
        )

        try:
            await pg.mark_status(doc.id, ContentStatus.PROCESSING)
            enriched = await self.enrich(doc, pg=pg)

            # Terminal docs go to review_queue, not pipeline_errors
            if enriched.status == ContentStatus.TERMINAL:
                await pg.mark_status(doc.id, ContentStatus.TERMINAL)
                reason = getattr(enriched, "skip_reason", None) or "empty_body"
                await pg.add_to_review_queue(
                    doc_id=doc.id,
                    reason=reason,
                    url=str(doc.url),
                    source=doc.source or "",
                )
                logger.info("Document %s moved to review_queue (%s)", doc.url, reason)
                return

            # Save raw document body if configured (only for non-terminal docs)
            if settings.save_raw_documents:
                out_dir = os.path.join("ingested_docs", str(doc.source or "unknown"))
                os.makedirs(out_dir, exist_ok=True)
                with open(os.path.join(out_dir, f"{doc.id}.txt"), "w", encoding="utf-8") as f:
                    f.write(doc.body or "")

            enriched.enriched_at = datetime.utcnow()
            enriched.embedding_model = settings.embed_model or _EMBEDDING_MODEL
            await pg.upsert(enriched)
            # Ensure aq_tsvec is always in sync with answers_questions.
            # pg.upsert() computes it from a joined string; this UPDATE reads
            # directly from the stored JSONB so the tsvector is authoritative.
            if enriched.answers_questions:
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
                                {"doc_id": str(enriched.id)},
                            )
                except Exception as _tsvec_exc:
                    logger.warning("aq_tsvec refresh failed for %s: %s", enriched.id, _tsvec_exc)
            logger.debug(
                "DB upsert: doc_id=%s url=%s status=%s topics=%s sentiment=%s embedding=%s",
                enriched.id,
                enriched.url,
                enriched.status,
                enriched.topics[:3],
                enriched.sentiment,
                enriched.embedding_model,
            )
            dump_step(
                doc.id,
                "04_db_upsert",
                {
                    "id": str(enriched.id),
                    "url": enriched.url,
                    "status": enriched.status.value if enriched.status else None,
                    "enrichment_version": enriched.enrichment_version,
                },
            )

            # Step 05 — embedding
            embed_text = build_embed_text(
                enriched.title,
                enriched.summary,
                enriched.answers_questions or [],
                enriched.body or "",
                enriched.embed_summary,
            )
            full_vectors: list[list[float]] = []
            async with httpx.AsyncClient() as http_client:
                vectors = await embed_batch(
                    http_client,
                    [embed_text],
                    full_out=full_vectors if settings.embed_store_full_vector else None,
                )
            if vectors:
                await pg.set_embedding(enriched.id, vectors[0])
                if full_vectors:
                    await pg.set_embedding_full(enriched.id, full_vectors[0])
                # Record "model:dims" after the actual vector is known.
                # This captures MRL truncation (OpenAI/Qwen with EMBED_DIMENSIONS)
                # and fixed-dim models (local/ollama/custom) accurately.
                actual_label = f"{settings.embed_model or _EMBEDDING_MODEL}:{len(vectors[0])}"
                enriched.embedding_model = actual_label
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
                            {"m": actual_label, "id": str(enriched.id)},
                        )
                        await _s.commit()
                except Exception as _em_exc:
                    logger.warning("embedding_model label update failed: %s", _em_exc)
                dump_step(
                    doc.id,
                    "05_embedding",
                    {
                        "doc_id": str(enriched.id),
                        "vector_length": len(vectors[0]),
                        "model": actual_label,
                    },
                )
            else:
                dump_step(
                    doc.id,
                    "05_embedding",
                    {
                        "doc_id": str(enriched.id),
                        "vector_length": 0,
                        "model": settings.embed_model or _EMBEDDING_MODEL,
                        "skipped": True,
                    },
                )

            # Step 06 — relationships (inverted-index SQL, O(k) not O(n)).
            # Edges are an enhancement: a failure here (e.g. Postgres-only SQL
            # on SQLite) must not mark an otherwise-enriched doc as failed.
            if getattr(pg, "_is_sqlite", False):
                logger.debug(
                    "edge building skipped for %s (relationship SQL is Postgres-only)",
                    doc.url,
                )
            else:
                try:
                    edge_count = await add_edges_for_doc(pg._engine, str(enriched.id))
                    dump_step(doc.id, "06_relationships", {"edge_count": edge_count})
                except Exception as exc:
                    logger.warning(
                        "edge building failed for %s (doc kept, no edges): %s", doc.url, exc
                    )

        except Exception:
            logger.exception("enrich_and_persist failed for %s", doc.url)
            await pg.mark_status(doc.id, ContentStatus.FAILED)


# ── Private helpers ───────────────────────────────────────────────────────────


def _save_raw_document(doc: ContentDocument) -> None:
    """
    Persist raw document body to disk when ``save_raw_documents`` is enabled.

    Writes to ``./ingested_docs/<source>/<doc_id>.txt``.
    Directory is auto-created. Silently ignores errors to avoid disrupting
    the enrichment pipeline.
    """
    from dewie.config import settings

    if not settings.save_raw_documents:
        return

    import os

    out_dir = os.path.join("ingested_docs", str(doc.source))
    os.makedirs(out_dir, exist_ok=True)
    try:
        with open(os.path.join(out_dir, f"{doc.id}.txt"), "w", encoding="utf-8") as f:
            f.write(doc.body or "")
    except Exception:
        logger.exception("Failed to save raw document %s", doc.id)


def _repair_json(text: str) -> str:
    """
    Best-effort repair of malformed LLM JSON output.

    Handles the three most common failure modes from long-context LLM responses:

    1. **Truncated / EOF** — response cut off mid-array due to token limit.
       Strategy: find the last complete top-level value, close any open
       brackets/braces so the JSON is syntactically valid.

    2. **Trailing commas** — LLM outputs JS-style ``[1, 2, 3,]`` or ``{"a":1,}``.
       Strategy: strip trailing commas before closing brackets/braces.

    3. **Missing commas between array items** — rare; not attempted (too risky
       to repair without misidentifying string content as separators).

    Returns the repaired string (may still be invalid if damage is severe).
    """
    import re as _re

    # Strip ASCII control characters (0x00-0x1F except \t \n \r) that break JSON parsers
    text = _re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

    # Strip trailing commas before ] or }
    text = _re.sub(r",\s*([\]}])", r"\1", text)

    # If the JSON looks truncated (no closing }), try to close open structures.
    # Walk the string tracking open brackets/braces; build a closing suffix.
    open_stack: list[str] = []
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in "{[":
            open_stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if open_stack and open_stack[-1] == ch:
                open_stack.pop()

    if open_stack:
        # Strip any trailing comma before we append closers
        text = _re.sub(r",\s*$", "", text.rstrip())
        text += "".join(reversed(open_stack))

    return text


def _parse_extraction_result(raw: str) -> ExtractionResult:
    """
    Parse a backend response string into an ``ExtractionResult``.

    Attempts multiple strategies in order:
    1. Direct parse (fast path — well-formed responses)
    2. Brace-extraction (strip prose wrapping)
    3. JSON repair (trailing commas, truncated arrays from token limits)

    Args:
        raw: Raw text response from the backend.

    Returns:
        A validated ``ExtractionResult`` instance.

    Raises:
        ValueError: If all parse strategies fail.
    """
    text = raw.strip()

    # Strip markdown code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    last_exc: Exception | None = None

    # Strategy 1: direct parse
    try:
        return ExtractionResult.model_validate_json(text)
    except (json.JSONDecodeError, ValueError) as exc:
        last_exc = exc

    # Strategy 2: brace-extraction (strip leading/trailing prose)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return ExtractionResult.model_validate_json(candidate)
        except (json.JSONDecodeError, ValueError) as exc:
            last_exc = exc
            text = text[start:]

        # Strategy 2b: repair up to the last "}" (handles partial trailing objects
        # that appear outside an already-closed array after token-limit truncation)
        repaired_candidate = _repair_json(candidate)
        if repaired_candidate != candidate:
            try:
                return ExtractionResult.model_validate_json(repaired_candidate)
            except (json.JSONDecodeError, ValueError) as exc:
                last_exc = exc

    # Strategy 3: repair then parse (trailing commas, truncated arrays)
    repaired = _repair_json(text)
    if repaired != text:
        try:
            return ExtractionResult.model_validate_json(repaired)
        except (json.JSONDecodeError, ValueError) as exc:
            last_exc = exc

    # Strategy 4: drop the optional alternate_terms field if it's what's corrupt.
    # alternate_terms is always last in the schema; truncation there shouldn't
    # kill the whole parse.  Strip it and retry with an empty array.
    import re as _re

    stripped = _re.sub(
        r',?\s*"alternate_terms"\s*:\s*\[.*',
        "",
        repaired,
        flags=_re.DOTALL,
    )
    if stripped != repaired:
        stripped = _repair_json(stripped)
        try:
            result = ExtractionResult.model_validate_json(stripped)
            result.alternate_terms = []  # type: ignore[attr-defined]
            logger.debug("_parse_extraction_result: alternate_terms dropped (truncated)")
            return result
        except (json.JSONDecodeError, ValueError):
            pass

    raise ValueError(
        f"Could not parse backend response into ExtractionResult. "
        f"Underlying error: {last_exc}. "
        f"Response (first 300 chars): {raw[:300]}"
    ) from last_exc


def _apply_result_to_doc(doc: ContentDocument, result: ExtractionResult) -> None:
    """
    Populate enrichment fields on ``doc`` from a parsed ``ExtractionResult``.

    Only non-empty / non-None result fields overwrite existing doc values.
    This preserves any fields that were set by a previous (partial) enrichment
    pass and allows hybrid workflows where different backends contribute
    different fields.

    Args:
        doc:    Document to update in-place.
        result: Parsed extraction result from the backend.
    """
    if result.document_type is not None:
        # Validate against DocumentType enum; fall back to OTHER on unknown value
        try:
            doc.document_type = DocumentType(result.document_type)
        except ValueError:
            doc.document_type = DocumentType.OTHER

    if result.keywords:
        doc.keywords = result.keywords

    if result.themes:
        doc.themes = result.themes
        # Also populate topics from themes if topics is empty (backward compat)
        if not doc.topics:
            doc.topics = result.themes

    if result.entities:
        doc.entities = [e.text for e in result.entities]

    if result.summary:
        doc.summary = result.summary[:MAX_SUMMARY_CHARS]

    if result.enrichment_quality_score is not None:
        doc.enrichment_quality_score = result.enrichment_quality_score

    if result.sentiment is not None:
        doc.sentiment = result.sentiment

    if result.tone:
        doc.tone = result.tone

    if result.author:
        doc.author = result.author

    if result.reading_level:
        try:
            doc.reading_level = ReadingLevel(result.reading_level)
        except ValueError:
            pass

    if result.language:
        doc.language = result.language

    if result.answers_questions:
        doc.answers_questions = result.answers_questions

    if result.embed_summary:
        doc.embed_summary = result.embed_summary[:MAX_EMBED_SUMMARY_CHARS]

    if result.alternate_terms:
        doc.alternate_terms = result.alternate_terms[:30]  # cap at 30 terms


async def _run_dual_pass(doc: ContentDocument) -> None:
    """Override AQ and KE results with dedicated focused LLM calls.

    Runs after single-pass extraction so doc already has summary/context.
    Failures are non-fatal — single-pass results are kept as fallback.
    """
    from dewie.pipeline import extract_ke, generate_aq

    title = doc.title or ""
    summary = doc.summary or ""
    body = doc.body or ""

    try:
        aq = await generate_aq(None, title, summary, body)  # type: ignore[arg-type]
        if aq:
            doc.answers_questions = aq
            logger.debug("dual_pass AQ: %d questions for %s", len(aq), doc.url)
    except Exception as exc:
        logger.warning("dual_pass AQ failed for %s (keeping single-pass result): %s", doc.url, exc)

    try:
        ke = await extract_ke(None, title, summary, body)  # type: ignore[arg-type]
        if ke.get("keywords"):
            doc.keywords = ke["keywords"]
        if ke.get("entities"):
            doc.entities = ke["entities"]
        logger.debug(
            "dual_pass KE: %d keywords, %d entities for %s",
            len(doc.keywords or []),
            len(doc.entities or []),
            doc.url,
        )
    except Exception as exc:
        logger.warning("dual_pass KE failed for %s (keeping single-pass result): %s", doc.url, exc)
