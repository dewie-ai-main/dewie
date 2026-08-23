# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
paywall_handler.py — Tiered handling for paywalled documents.

Provides:
  - classify_paywall_document(doc) — categorize a paywalled document
  - apply_terminal_status(doc, status) — mark terminal documents

Classification logic:
  - normal        — no paywall detected
  - terminal_no_body — paywalled with no body text → mark terminal
  - terminal_stub   — paywalled with short body (<500 chars) → mark terminal
  - enrich_normal   — paywalled with substantial body (>=500 chars) → enrich as normal

Usage:
    from dewie.ingestion.paywall_handler import classify_paywall_document, apply_terminal_status

    status = classify_paywall_document(doc)
    if status.startswith("terminal_"):
        apply_terminal_status(doc, status)
"""

from __future__ import annotations

import logging

from dewie.models.content import ContentDocument

log = logging.getLogger(__name__)


def classify_paywall_document(doc: ContentDocument) -> str:
    """
    Classify a document based on paywall status and body content.

    Returns one of:
      - 'normal'        — no paywall detected
      - 'terminal_no_body' — paywalled, no body text
      - 'terminal_stub'   — paywalled, body < 500 chars
      - 'enrich_normal'   — paywalled, body >= 500 chars
    """
    if not doc.paywall_detected:
        return "normal"

    body = doc.body or ""
    body_len = len(body)

    if body_len == 0 or body.strip() == "":
        return "terminal_no_body"
    elif body_len < 500:
        return "terminal_stub"
    else:
        return "enrich_normal"


def apply_terminal_status(doc: ContentDocument, status: str) -> None:
    """
    Apply terminal status to a document based on its classification.

    Terminal documents are marked with status='terminal' and a skip_reason.
    """
    if status == "terminal_no_body":
        doc.status = "terminal"
        doc.skip_reason = "paywall_no_body"
    elif status == "terminal_stub":
        doc.status = "terminal"
        doc.skip_reason = "stub"
    elif status == "enrich_normal":
        log.warning("Paywalled doc with body > 500 chars will be enriched.")
