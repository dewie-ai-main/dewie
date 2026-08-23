# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from dewie.config import settings

logger = logging.getLogger(__name__)

_VALID_SOURCE_TYPES = frozenset({"sqlite", "postgres", "mcp"})


def _parse_sources_payload(raw: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning("Invalid PUBLIC_SOURCES_JSON payload: %s", exc)
        return []

    if not isinstance(parsed, list):
        logger.warning("PUBLIC_SOURCES_JSON must be a JSON array")
        return []

    return [item for item in parsed if isinstance(item, dict)]


def _validate_source(item: dict[str, Any]) -> dict[str, Any] | None:
    name = str(item.get("name", "")).strip()
    source_type = str(item.get("type", "")).strip()
    if not name or source_type not in _VALID_SOURCE_TYPES:
        return None

    config = item.get("config")
    if not isinstance(config, dict):
        config = {}

    return {
        "name": name,
        "type": source_type,
        "config": config,
        "enabled": bool(item.get("enabled", True)),
    }


_DEV_POSTGRES_DSN = "postgresql+asyncpg://dewie:dewie@localhost:5432/dewie"


def build_local_source() -> dict[str, Any] | None:
    """Return a catalog entry for this instance's own database, or None if not detectable.

    Seeds a 'local' entry so agents can always discover the running node without
    manual registration. Skipped when the DSN is the dev-only placeholder.
    """
    dewie_db = (settings.dewie_db or "").strip()
    if dewie_db:
        return {
            "name": "local",
            "type": "sqlite",
            "config": {"filepath": dewie_db},
            "enabled": True,
        }

    postgres_dsn = (settings.postgres_dsn or "").strip()
    if postgres_dsn and postgres_dsn != _DEV_POSTGRES_DSN:
        return {
            "name": "local",
            "type": "postgres",
            "config": {"dsn": postgres_dsn},
            "enabled": True,
        }

    return None


def load_public_sources_defaults() -> list[dict[str, Any]]:
    """Load startup source defaults from file + env, merged by source name.

    Merge precedence:
    1) file defaults (base)
    2) public_sources_json override (wins on same name)
    """
    merged: dict[str, dict[str, Any]] = {}

    file_path = Path(settings.public_sources_file).expanduser()
    if file_path.exists():
        try:
            file_payload = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(file_payload, list):
                for raw in file_payload:
                    if isinstance(raw, dict):
                        validated = _validate_source(raw)
                        if validated is not None:
                            merged[validated["name"]] = validated
            else:
                logger.warning("public sources file must contain a JSON array: %s", file_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed reading public sources file %s: %s", file_path, exc)

    env_raw = settings.public_sources_json.strip()
    if env_raw:
        for raw in _parse_sources_payload(env_raw):
            validated = _validate_source(raw)
            if validated is not None:
                merged[validated["name"]] = validated

    return list(merged.values())
