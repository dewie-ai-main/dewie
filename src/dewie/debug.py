# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Debug tracing for the enrichment pipeline.

Set DEWIE_DEBUG=1 to enable. Dumps intermediate state to /tmp/dewie_debug/{doc_id}/.
Each step writes a JSON file: 01_body_load.json, 02_llm_extraction.json, etc.
"""

import json
import logging
import os
from pathlib import Path
from uuid import UUID

DEBUG = os.environ.get("DEWIE_DEBUG", "0") == "1"
DEBUG_DIR = Path("/tmp/dewie_debug")
log = logging.getLogger(__name__)


def dump_step(doc_id: str | UUID, step: str, data: dict) -> None:
    """Write a debug snapshot for one pipeline step. No-op if DEBUG=False."""
    if not DEBUG:
        return
    try:
        d = DEBUG_DIR / str(doc_id)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{step}.json").write_text(json.dumps(data, indent=2, default=str))
    except Exception as e:
        log.debug("debug dump failed: %s", e)
