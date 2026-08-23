#!/usr/bin/env python3
"""
Stage 3: Enrichment test — force-run the enrichment pipeline on a specific doc.

Loads a doc from Postgres by ID (or picks the newest test:enrich doc),
calls MetadataProcessor.enrich() directly (bypassing the API and queue),
times each phase, and prints a full field audit.

Usage:
    # Enrich a specific doc by ID:
    python scripts/test_stage3_enrich.py --doc-id <uuid>

    # Enrich the doc output by Stage 2 (reads artifact):
    python scripts/test_stage3_enrich.py --run-id my-run

    # Enrich the most recent test:enrich doc in the DB:
    python scripts/test_stage3_enrich.py --latest

    # Re-enrich even if already READY (force):
    python scripts/test_stage3_enrich.py --doc-id <uuid> --force

Phases timed separately:
  1. DB load + body read
  2. LLM extraction (main pass)
  3. Dual-pass AQ / KE  (if enrichment_mode=dual_pass)
  4. Embedding
  5. DB persist
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

ARTIFACT_DIR = Path("data/test_artifacts/stage2_inject")

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"


def check(label: str, condition: bool, detail: str = "") -> bool:
    marker = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{marker}] {label}{suffix}")
    return condition


def warn(label: str, detail: str = "") -> None:
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{WARN}] {label}{suffix}")


async def _resolve_doc_id(args, pg) -> uuid.UUID | None:
    if args.doc_id:
        return uuid.UUID(args.doc_id)

    if args.run_id:
        artifact = ARTIFACT_DIR / f"{args.run_id}.json"
        if not artifact.exists():
            print(f"  [{FAIL}] artifact not found: {artifact}")
            return None
        data = json.loads(artifact.read_text())
        return uuid.UUID(data["doc_id"])

    if args.latest:
        # Grab most recently inserted test:enrich doc
        from sqlalchemy import text

        async with pg._engine.begin() as conn:
            row = await conn.execute(
                text(
                    "SELECT id FROM documents WHERE corpus_id = 'test:enrich' "
                    "ORDER BY ingested_at DESC LIMIT 1"
                )
            )
            result = row.fetchone()
        if not result:
            print(f"  [{FAIL}] no test:enrich docs in database")
            return None
        return uuid.UUID(str(result[0]))

    return None


async def run(args: argparse.Namespace) -> int:
    from dewie.config import settings
    from dewie.enrichment.processor import MetadataProcessor
    from dewie.enrichment.registry import BackendRegistry
    from dewie.enrichment.router import EnrichmentRouter
    from dewie.models.content import ContentStatus
    from dewie.pipeline import embed_batch
    from dewie.storage.body_store import load_body
    from dewie.storage.postgres import PostgresClient

    pg = PostgresClient()
    ok = True

    # ── Phase 0: resolve doc ──────────────────────────────────────────────────
    doc_id = await _resolve_doc_id(args, pg)
    if not doc_id:
        print("Provide --doc-id <uuid>, --run-id <id>, or --latest")
        return 1

    print(f"\nStage 3: Enrich\n  doc_id: {doc_id}\n")

    # ── Phase 1: load doc + body ──────────────────────────────────────────────
    t0 = time.monotonic()
    doc = await pg.get_by_id(doc_id)
    load_ms = (time.monotonic() - t0) * 1000

    ok &= check("doc found in postgres", doc is not None, f"{load_ms:.0f}ms")
    if not doc:
        return 1

    print(f"  title  : {doc.title[:80]}")
    print(f"  url    : {doc.url}")
    print(f"  status : {doc.status.value}\n")

    if doc.status == ContentStatus.READY and not args.force:
        warn("doc is already READY — use --force to re-enrich")
        print(f"\n  summary    : {(doc.summary or '')[:200]}")
        print(f"  keywords   : {doc.keywords[:8]}")
        print(f"  aq_count   : {len(doc.answers_questions or [])}")
        return 0

    # Load body from body_store (preferred) or DB body_text column
    body = load_body(doc.id)
    if body:
        doc.body = body
        check("body loaded from body_store", True, f"{len(body)} chars")
    else:
        warn("body_store miss — loading from postgres body_text")
        from sqlalchemy import text

        async with pg._engine.begin() as conn:
            row = await conn.execute(
                text("SELECT body_text FROM documents WHERE id = :id"),
                {"id": str(doc.id)},
            )
            result = row.fetchone()
        if result and result[0]:
            doc.body = result[0]
            check("body loaded from postgres body_text", True, f"{len(doc.body)} chars")
        else:
            ok &= check("body available", False, "empty in both body_store and body_text column")
            return 1

    ok &= check("body non-empty", bool(doc.body and len(doc.body.strip()) > 100))

    # Reset status for fresh enrich
    doc.status = ContentStatus.PENDING

    # ── Phase 2: build processor ──────────────────────────────────────────────
    registry = BackendRegistry.from_config(settings)
    router = EnrichmentRouter.from_config(settings, registry)
    processor = MetadataProcessor(
        router=router,
        registry=registry,
        max_retries=settings.enrichment_max_retries,
    )

    selected_backend = router.select(doc)
    print(f"  backend: {selected_backend.name}")
    print(f"  mode   : {settings.enrichment_mode}\n")

    # ── Phase 3: LLM extraction (timed) ──────────────────────────────────────
    print("  Running enrichment...")
    phase_times: dict[str, float] = {}

    llm_start = time.monotonic()
    enriched = await processor.enrich(doc, pg=None)
    phase_times["enrich_total"] = time.monotonic() - llm_start

    ok &= check(
        "enrichment succeeded",
        enriched.status == ContentStatus.READY,
        f"status={enriched.status.value}",
    )

    if enriched.status != ContentStatus.READY:
        print(f"\n  [{FAIL}] Enrichment failed — doc status: {enriched.status.value}")
        return 1

    # ── Phase 4: field audit ──────────────────────────────────────────────────
    print(f"\n  Timing: {phase_times['enrich_total']:.1f}s total\n")
    print("  Field audit:")

    fields: list[tuple[str, Any, bool]] = [
        ("summary", enriched.summary, bool(enriched.summary and len(enriched.summary) > 20)),
        ("embed_summary", enriched.embed_summary, bool(enriched.embed_summary)),
        ("document_type", enriched.document_type, enriched.document_type is not None),
        ("keywords", enriched.keywords, len(enriched.keywords or []) >= 3),
        ("entities", enriched.entities, True),  # optional
        ("topics", enriched.topics, True),       # optional
        ("sentiment", enriched.sentiment, enriched.sentiment is not None),
        ("tone", enriched.tone, bool(enriched.tone)),
        ("reading_level", enriched.reading_level, enriched.reading_level is not None),
        ("language", enriched.language, bool(enriched.language)),
        ("answers_questions", enriched.answers_questions, len(enriched.answers_questions or []) >= 1),
    ]

    for name, value, required in fields:
        if isinstance(value, list):
            display = f"{len(value)} items: {value[:3]}"
            present = bool(value)
        elif value is not None and value != "":
            display = repr(str(value)[:60])
            present = True
        else:
            display = "MISSING"
            present = False

        if required:
            ok &= check(name, present, display)
        else:
            if present:
                print(f"  [{PASS}] {name}  ({display})")
            else:
                warn(name, "optional, empty")

    # ── Phase 5: persist ──────────────────────────────────────────────────────
    print("\n  Persisting to postgres...")
    t_persist = time.monotonic()
    await pg.upsert(enriched)
    persist_ms = (time.monotonic() - t_persist) * 1000
    ok &= check("upserted enriched doc", True, f"{persist_ms:.0f}ms")

    # Verify status in DB
    refetched = await pg.get_by_id(doc.id)
    ok &= check(
        "DB status is READY after persist",
        refetched is not None and refetched.status == ContentStatus.READY,
    )

    # ── Phase 6: embedding (optional, skip if no embed server) ───────────────
    print("\n  Embedding (may skip if server unavailable)...")
    try:
        from dewie.pipeline import build_embed_text

        embed_text = build_embed_text(
            enriched.title,
            enriched.summary or "",
            enriched.answers_questions or [],
            enriched.body or "",
            embed_summary=enriched.embed_summary,
        )
        import httpx as _httpx
        t_embed = time.monotonic()
        async with _httpx.AsyncClient() as http:
            vectors = await embed_batch(http, [embed_text])
        embed_ms = (time.monotonic() - t_embed) * 1000

        if vectors:
            await pg.set_embedding(doc.id, vectors[0])
            ok &= check("embedding generated and stored", True, f"dim={len(vectors[0])}, {embed_ms:.0f}ms")
        else:
            warn("embedding skipped", "embed_batch returned None (server may be unavailable)")
    except Exception as e:
        warn("embedding phase error", str(e)[:80])

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  doc_id  : {doc.id}")
    print(f"  summary : {(enriched.summary or '')[:200]}")
    print(f"  keywords: {(enriched.keywords or [])[:8]}")
    print(f"  AQ      : {len(enriched.answers_questions or [])} questions")
    print(f"  total   : {phase_times['enrich_total']:.1f}s\n")

    return 0 if ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--doc-id", help="UUID of doc to enrich")
    group.add_argument("--run-id", help="run-id from Stage 2 artifact")
    group.add_argument("--latest", action="store_true", help="pick the newest test:enrich doc")
    parser.add_argument("--force", action="store_true", help="re-enrich even if already READY")
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args))  )
