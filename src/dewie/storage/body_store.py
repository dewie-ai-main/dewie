# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Flat-file body store.

Writes raw extracted text to disk alongside the DB so we can rebuild
the corpus from scratch if ingestion logic or enrichment changes.

Layout:
    {data_dir}/bodies/{doc_id[:2]}/{doc_id}.txt   (2-char prefix sharding)

The shard prefix avoids hitting filesystem inode limits at 100K+ files.

The base directory is resolved at call time from settings.data_dir, defaulting
to ./data relative to the process working directory.  Set DEWIE_DATA_DIR in
the environment (or .env) to pin it — recommended for Docker deployments.
"""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import UUID

log = logging.getLogger(__name__)

# Override for testing — monkeypatch this to redirect I/O without touching settings.
BODIES_DIR: Path | None = None


def _bodies_dir() -> Path:
    """Resolve the bodies directory from settings (lazy, so cwd is stable)."""
    if BODIES_DIR is not None:
        return BODIES_DIR
    from dewie.config import settings

    base = Path(settings.data_dir) if settings.data_dir else Path.cwd() / "data"
    return base / "bodies"


# Module-level alias for callers that read it directly (e.g. health checks).
# Re-evaluated each call so DEWIE_DATA_DIR changes take effect without reload.
def bodies_dir() -> Path:
    return _bodies_dir()


def save_body(doc_id: UUID | str, body: str) -> None:
    """Write body text to disk. Silently skips empty bodies."""
    if not body or not body.strip():
        return
    doc_id = str(doc_id)
    shard = doc_id[:2]
    path = _bodies_dir() / shard / f"{doc_id}.txt"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    except Exception as exc:
        log.warning("body_store: failed to write %s: %s", path, exc)


def load_body(doc_id: UUID | str) -> str | None:
    """Read body text from disk. Returns None if not found."""
    doc_id = str(doc_id)
    shard = doc_id[:2]
    path = _bodies_dir() / shard / f"{doc_id}.txt"
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except Exception as exc:
        log.warning("body_store: failed to read %s: %s", path, exc)
        return None


def body_exists(doc_id: UUID | str) -> bool:
    doc_id = str(doc_id)
    return (_bodies_dir() / doc_id[:2] / f"{doc_id}.txt").exists()


def delete_body(doc_id: UUID | str) -> None:
    """Delete body text from disk. Silently ignores missing files."""
    doc_id = str(doc_id)
    path = _bodies_dir() / doc_id[:2] / f"{doc_id}.txt"
    try:
        path.unlink(missing_ok=True)
    except Exception as exc:
        log.warning("body_store: failed to delete %s: %s", path, exc)
