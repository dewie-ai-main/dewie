#!/usr/bin/env python3
"""
Podcast episode ingestion CLI — the entrypoint docker-compose.podcast.yml runs.

Fetches a podcast RSS/Atom feed, transcribes new episodes with Whisper
(openai-whisper or faster-whisper, whichever is installed), and writes
transcripts to the documents table with status=pending so the enrichment
worker picks them up.

Usage:
    python scripts/ingest_podcast.py FEED_URL [--limit N] [--model SIZE]
        [--corpus-id UUID] [--since ISO_DATE] [--dry-run]

Database: reads DEWIE_DSN (or POSTGRES_DSN, or discrete POSTGRES_* vars)
from the environment. --dry-run transcribes without touching the database.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import uuid
from datetime import UTC, datetime


def _build_dsn() -> str | None:
    """Resolve a sync-driver DSN from the environment."""
    dsn = os.environ.get("DEWIE_DSN") or os.environ.get("POSTGRES_DSN")
    if dsn:
        # Tolerate async-flavoured URLs (e.g. postgresql+asyncpg://) — psycopg2
        # only understands the plain scheme.
        return dsn.replace("postgresql+asyncpg://", "postgresql://", 1)
    host = os.environ.get("POSTGRES_HOST")
    if not host:
        return None
    user = os.environ.get("POSTGRES_USER", "dewie")
    password = os.environ.get("POSTGRES_PASSWORD", "")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "dewie")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"


def _parse_since(raw: str) -> datetime:
    dt = datetime.fromisoformat(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("feed_url", help="Podcast RSS/Atom feed URL")
    parser.add_argument(
        "--limit", type=int, default=10, help="Max episodes to process, newest first (default 10)"
    )
    parser.add_argument(
        "--model",
        default="base",
        help="Whisper model size: tiny/base/small/medium/large (default base)",
    )
    parser.add_argument(
        "--corpus-id", default=None, help="Corpus UUID to file episodes under (optional)"
    )
    parser.add_argument(
        "--since",
        type=_parse_since,
        default=None,
        metavar="ISO_DATE",
        help="Skip episodes published before this date (e.g. 2026-01-01)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Transcribe but skip all database access"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    corpus_id: str | None = None
    if args.corpus_id:
        try:
            corpus_id = str(uuid.UUID(args.corpus_id))
        except ValueError:
            print(
                f"error: --corpus-id must be a UUID, got {args.corpus_id!r} "
                "(episodes are grouped by feed host in `source` automatically; "
                "only pass --corpus-id to file them under an existing corpus)",
                file=sys.stderr,
            )
            return 2

    from dewie.ingestion.podcast import (
        ingest_podcast_feed,
        is_podcast_transcription_available,
    )

    if not is_podcast_transcription_available():
        print(
            "error: no Whisper backend installed — pip install faster-whisper (or openai-whisper)",
            file=sys.stderr,
        )
        return 2

    pg = None
    if not args.dry_run:
        dsn = _build_dsn()
        if not dsn:
            print(
                "error: no database configured — set DEWIE_DSN, POSTGRES_DSN, "
                "or POSTGRES_HOST/USER/PASSWORD/DB (or pass --dry-run)",
                file=sys.stderr,
            )
            return 2
        import psycopg2

        pg = psycopg2.connect(dsn)

    try:
        results = asyncio.run(
            ingest_podcast_feed(
                args.feed_url,
                pg,
                limit=args.limit,
                model_size=args.model,
                dry_run=args.dry_run,
                corpus_id=corpus_id,
                since=args.since,
            )
        )
    finally:
        if pg is not None:
            pg.close()

    if not results:
        print("No episodes matched the filters (feed empty, or all older than --since).")
        return 0

    counts: dict[str, int] = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        line = f"  [{r['status']:8}] {r['title'][:70]}"
        if r.get("words"):
            line += f" ({r['words']} words)"
        if r.get("error"):
            line += f" — {r['error']}"
        print(line)

    summary = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    print(f"\n{len(results)} episode(s): {summary}")

    # Fail loudly only when every non-skipped episode failed.
    attempted = len(results) - counts.get("skipped", 0)
    if attempted > 0 and counts.get("failed", 0) == attempted:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
