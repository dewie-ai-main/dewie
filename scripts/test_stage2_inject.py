#!/usr/bin/env python3
"""
Stage 2: Inject test — create a synthetic doc and persist it directly.

Skips URL fetching entirely. Builds a realistic ContentDocument with a
known body, writes it to Postgres + body_store under corpus_id='test:enrich',
and dumps an artifact so Stage 3 can enrich it.

The test doc URL is deterministic:
    https://dewie-test.internal/enrichment-test/<run-id>

This means re-running Stage 2 with the same --run-id is idempotent (upserts).

Usage:
    python scripts/test_stage2_inject.py [--run-id my-run]

    # Use a specific doc length for stress testing:
    python scripts/test_stage2_inject.py --size medium
    python scripts/test_stage2_inject.py --size long

Artifact: data/test_artifacts/stage2_inject/<run_id>.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from dewie.models.content import ContentDocument, ContentStatus
from dewie.storage.body_store import save_body
from dewie.storage.postgres import PostgresClient

ARTIFACT_DIR = Path("data/test_artifacts/stage2_inject")
TEST_CORPUS = "test:enrich"

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
WARN = "\033[33mWARN\033[0m"


def warn(label: str, detail: str = "") -> None:
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{WARN}] {label}{suffix}")

# Sized synthetic bodies — realistic prose, not lorem ipsum
_PARA = (
    "Large language models have fundamentally changed how software engineers approach "
    "natural language processing tasks. Instead of hand-crafting feature pipelines, "
    "practitioners now fine-tune or prompt pre-trained transformers with billions of "
    "parameters. The trade-offs are non-trivial: inference cost scales with model size, "
    "context windows constrain document length, and hallucination rates vary across domains. "
    "Retrieval-augmented generation (RAG) addresses some limitations by grounding model "
    "outputs in a retrieved document corpus. The enrichment pipeline described here uses "
    "a hybrid approach: a large instruction-tuned model extracts structured metadata "
    "(summary, keywords, entities, reading level, sentiment) while a separate embedding "
    "model encodes a dense retrieval vector for approximate nearest-neighbour search. "
    "The dual-pass architecture separates factual extraction from question-answering, "
    "allowing each step to be swapped independently as better models become available. "
)

SIZES = {
    "stub": (_PARA[:300], "Stub — 300 chars"),
    "short": (_PARA * 2, "Short — ~1.5k chars"),
    "medium": (_PARA * 8, "Medium — ~6k chars"),
    "long": (_PARA * 25, "Long — ~18k chars"),
    "xlarge": (_PARA * 80, "XLarge — ~60k chars"),
}


def check(label: str, condition: bool, detail: str = "") -> bool:
    marker = PASS if condition else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{marker}] {label}{suffix}")
    return condition


async def run(run_id: str, size: str) -> int:
    body_text, size_label = SIZES[size]
    url = f"https://dewie-test.internal/enrichment-test/{run_id}"

    print(f"\nStage 2: Inject\n  run_id : {run_id}\n  size   : {size_label}\n  url    : {url}\n")

    doc = ContentDocument.from_url(url)
    doc.title = f"[TEST] Enrichment pipeline test — {size_label}"
    doc.source = "dewie-test"
    doc.corpus_id = TEST_CORPUS
    doc.status = ContentStatus.PENDING
    doc.body = body_text

    print(f"  doc_id : {doc.id}")
    print(f"  body   : {len(doc.body)} chars\n")

    pg = PostgresClient()
    ok = True

    # Write to DB
    t0 = time.monotonic()
    await pg.upsert(doc)
    db_ms = (time.monotonic() - t0) * 1000
    ok &= check("upserted to postgres", True, f"{db_ms:.0f}ms")

    # ON CONFLICT(url) keeps the existing id — re-read to get the canonical id
    from sqlalchemy import text as _text
    async with pg._engine.begin() as conn:
        row = await conn.execute(_text("SELECT id FROM documents WHERE url = :url"), {"url": url})
        result = row.fetchone()
    if not result:
        ok &= check("doc found by url after upsert", False)
        return 1
    doc_id = result[0]
    # Sync id back onto doc so artifact and body_store use the canonical id
    import uuid as _uuid
    doc.id = _uuid.UUID(str(doc_id))

    # Write body to flat file store (keyed by canonical id)
    save_body(doc.id, doc.body)
    ok &= check("body saved to body_store", True)

    # Persist body_text column as backup
    await pg.write_body_text(doc.id, doc.body)
    ok &= check("body_text written to postgres", True)

    # Verify
    fetched = await pg.get_by_id(doc.id)
    ok &= check("doc readable from postgres", fetched is not None)
    if fetched:
        # corpus_id is a UUID FK in the DB now; just check it's non-null
        if fetched.corpus_id:
            check("corpus_id set", True, str(fetched.corpus_id))
        else:
            warn("corpus_id not set (corpus may not exist in corpora table)")
        ok &= check("status is PENDING", fetched.status == ContentStatus.PENDING)
        ok &= check("title preserved", fetched.title == doc.title)

    # Write artifact
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "doc_id": str(doc.id),
        "run_id": run_id,
        "url": url,
        "corpus_id": TEST_CORPUS,
        "size": size,
        "body_chars": len(doc.body),
        "status": doc.status.value,
    }
    artifact_path = ARTIFACT_DIR / f"{run_id}.json"
    artifact_path.write_text(json.dumps(artifact, indent=2))

    print(f"\n  artifact: {artifact_path}")
    print(f"  doc_id for Stage 3: {doc.id}\n")

    return 0 if ok else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=str(uuid.uuid4())[:8])
    parser.add_argument(
        "--size",
        choices=list(SIZES),
        default="medium",
        help="Body size class (default: medium)",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(run(args.run_id, args.size)))
