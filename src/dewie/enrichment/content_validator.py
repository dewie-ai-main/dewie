# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Content quality validators for the Dewie ingest pipeline.

Two layers:
  1. Pre-enrichment (ContentValidator) — runs before LLM enrichment.
     Catches garbage, stubs, near-duplicate content, and low-signal text.
     Called from ingest.py before _enrich_batch().

  2. Post-enrichment (EnrichmentQualityChecker) — runs periodically on ready docs.
     Catches bad LLM output: generic AQs, mismatched embeddings, summary bleed.
     Used by scripts/run_enrichment_quality_check.py.

Usage (pre-enrichment):
    from dewie.enrichment.content_validator import ContentValidator, ValidationResult
    result = ContentValidator.validate(doc)
    if not result.ok:
        log.warning("Skipping %s: %s", doc.url, result.reason)
        continue

Usage (post-enrichment):
    from dewie.enrichment.content_validator import EnrichmentQualityChecker
    flags = EnrichmentQualityChecker.check(doc)
    for flag in flags:
        log.warning("Quality flag: doc=%s check=%s reason=%s", flag.doc_id, flag.check, flag.reason)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dewie.models.content import ContentDocument

# ── Defaults ─────────────────────────────────────────────────────────────────

MIN_BODY_CHARS = 200  # absolute floor — below this, not worth enriching
MIN_BODY_WORDS = 40  # word-count floor
MAX_NOISE_RATIO = 0.35  # fraction of non-word chars before we flag as noise
MAX_REPETITION_RATIO = 0.6  # fraction of repeated lines before we flag
MIN_ALPHA_RATIO = 0.50  # fraction of alphabetic chars in body
MAX_BODY_CHARS = 5_000_000  # hard cap — something is very wrong above this
MIN_TITLE_CHARS = 3  # minimum meaningful title

BOILERPLATE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^javascript\s+required",
        r"^please\s+enable\s+javascript",
        r"^access\s+denied",
        r"^403\s+forbidden",
        r"^404\s+not\s+found",
        r"^this\s+(page|site|content)\s+(requires|needs|uses)\s+javascript",
        r"^you\s+need\s+to\s+enable\s+javascript",
        r"^subscribe\s+to\s+(read|continue|access)",
        r"^sign\s+in\s+to\s+(read|continue|access|view)",
        r"^\s*\[\s*\.\.\.\s*\]\s*$",  # "[...]" stubs
        r"^\s*loading\.\.\.\s*$",
        r"^\s*please\s+wait\.\.\.\s*$",
    ]
]

GENERIC_TITLE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^untitled\s*$",
        r"^no\s+title\s*$",
        r"^home\s*[\|\-–]",
        r"^\s*$",
        r"^document\s*$",
        r"^page\s+\d+\s*$",
    ]
]


# ── Pre-enrichment validation ─────────────────────────────────────────────────


@dataclass
class ValidationResult:
    ok: bool
    reason: str = ""
    checks_failed: list[str] = field(default_factory=list)

    @classmethod
    def pass_(cls) -> ValidationResult:
        return cls(ok=True)

    @classmethod
    def fail(cls, reason: str, check: str = "") -> ValidationResult:
        return cls(ok=False, reason=reason, checks_failed=[check] if check else [])


class ContentValidator:
    """
    Pre-enrichment content quality gate.

    All checks are independent and configurable via keyword arguments.
    Default thresholds are conservative — tuned to catch obvious garbage
    without over-filtering.

    Usage:
        result = ContentValidator.validate(doc)
        result = ContentValidator.validate(doc, min_body_chars=500, check_boilerplate=False)
    """

    @classmethod
    def validate(
        cls,
        doc: ContentDocument,
        *,
        min_body_chars: int = MIN_BODY_CHARS,
        min_body_words: int = MIN_BODY_WORDS,
        max_noise_ratio: float = MAX_NOISE_RATIO,
        max_repetition_ratio: float = MAX_REPETITION_RATIO,
        min_alpha_ratio: float = MIN_ALPHA_RATIO,
        max_body_chars: int = MAX_BODY_CHARS,
        min_title_chars: int = MIN_TITLE_CHARS,
        check_boilerplate: bool = True,
        check_repetition: bool = True,
        check_noise: bool = True,
        check_title: bool = True,
    ) -> ValidationResult:
        """Run all enabled checks. Returns first failure or pass."""

        body: str = (getattr(doc, "body", None) or getattr(doc, "body_text", None) or "").strip()
        title: str = (getattr(doc, "title", None) or "").strip()
        url: str = str(getattr(doc, "url", None) or "")

        # ── Length checks ─────────────────────────────────────────────────────
        if len(body) < min_body_chars:
            return ValidationResult.fail(
                f"body too short ({len(body)} chars < {min_body_chars})",
                check="min_body_chars",
            )

        if len(body) > max_body_chars:
            return ValidationResult.fail(
                f"body suspiciously large ({len(body)} chars > {max_body_chars})",
                check="max_body_chars",
            )

        word_count = len(body.split())
        if word_count < min_body_words:
            return ValidationResult.fail(
                f"body too few words ({word_count} < {min_body_words})",
                check="min_body_words",
            )

        # ── Boilerplate detection ─────────────────────────────────────────────
        if check_boilerplate:
            first_200 = body[:200].strip()
            for pattern in BOILERPLATE_PATTERNS:
                if pattern.search(first_200):
                    return ValidationResult.fail(
                        f"boilerplate content detected: {first_200[:80]!r}",
                        check="boilerplate",
                    )

        # ── Alphabetic ratio ──────────────────────────────────────────────────
        if check_noise and len(body) > 0:
            alpha_count = sum(1 for c in body if c.isalpha())
            alpha_ratio = alpha_count / len(body)
            if alpha_ratio < min_alpha_ratio:
                return ValidationResult.fail(
                    f"low alphabetic ratio ({alpha_ratio:.2f} < {min_alpha_ratio}) — likely binary/encoded content",
                    check="alpha_ratio",
                )

        # ── Noise ratio ───────────────────────────────────────────────────────
        if check_noise and len(body) > 0:
            word_chars = sum(1 for c in body if c.isalnum() or c.isspace())
            noise_ratio = 1.0 - (word_chars / len(body))
            if noise_ratio > max_noise_ratio:
                return ValidationResult.fail(
                    f"high noise ratio ({noise_ratio:.2f} > {max_noise_ratio}) — likely HTML/script bleed",
                    check="noise_ratio",
                )

        # ── Repetition detection ──────────────────────────────────────────────
        if check_repetition:
            lines = [l.strip() for l in body.splitlines() if l.strip()]
            if len(lines) >= 10:
                unique_lines = set(lines)
                repetition_ratio = 1.0 - (len(unique_lines) / len(lines))
                if repetition_ratio > max_repetition_ratio:
                    return ValidationResult.fail(
                        f"high line repetition ({repetition_ratio:.2f} > {max_repetition_ratio}) — likely nav/menu bleed",
                        check="repetition",
                    )

        # ── Title checks ──────────────────────────────────────────────────────
        if check_title:
            if len(title) < min_title_chars:
                return ValidationResult.fail(
                    f"title too short or missing ({title!r})",
                    check="title_length",
                )
            for pattern in GENERIC_TITLE_PATTERNS:
                if pattern.search(title):
                    return ValidationResult.fail(
                        f"generic/placeholder title detected ({title!r})",
                        check="generic_title",
                    )

        return ValidationResult.pass_()

    @classmethod
    def validate_many(
        cls,
        docs: list[ContentDocument],
        **kwargs,
    ) -> tuple[list[ContentDocument], list[tuple[ContentDocument, ValidationResult]]]:
        """
        Validate a list of documents.
        Returns (passed_docs, [(rejected_doc, result), ...]).
        """
        passed = []
        rejected = []
        for doc in docs:
            result = cls.validate(doc, **kwargs)
            if result.ok:
                passed.append(doc)
            else:
                rejected.append((doc, result))
        return passed, rejected


# ── Post-enrichment quality checks ───────────────────────────────────────────


@dataclass
class QualityFlag:
    doc_id: str
    check: str
    reason: str
    severity: str = "warning"  # "warning" | "error"


GENERIC_AQ_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"^what\s+is\s+this\s+(article|document|page|text|post)\s+about",
        r"^what\s+does\s+this\s+(article|document|page|text|post)\s+(say|discuss|cover|contain|describe)",
        r"^(can\s+you\s+)?summarize\s+this",
        r"^tell\s+me\s+about\s+this",
        r"^what\s+are\s+the\s+(main\s+)?(topics?|points?|themes?|subjects?)\s+(in|of|covered)",
        r"^what\s+information\s+is\s+(provided|given|contained)",
    ]
]


class EnrichmentQualityChecker:
    """
    Post-enrichment quality checker.

    Runs against already-enriched (status=ready) documents to catch
    bad LLM output that passed the null checks but is still low quality.

    Usage:
        flags = EnrichmentQualityChecker.check(doc)
        for flag in flags: ...
    """

    @classmethod
    def check(
        cls,
        doc: ContentDocument,
        *,
        min_summary_chars: int = 50,
        max_summary_chars: int = 5000,
        min_aq_count: int = 2,
        max_aq_count: int = 20,
        min_aq_length: int = 15,
        check_generic_aq: bool = True,
        check_duplicate_aq: bool = True,
        check_summary_bleed: bool = True,
    ) -> list[QualityFlag]:
        """Run all post-enrichment checks. Returns list of flags (empty = clean)."""
        flags: list[QualityFlag] = []
        doc_id = str(getattr(doc, "id", None) or "unknown")
        summary: str = (getattr(doc, "summary", None) or "").strip()
        embed_summary: str = (getattr(doc, "embed_summary", None) or "").strip()
        aq: list[str] = getattr(doc, "answers_questions", None) or []
        body: str = (getattr(doc, "body", None) or getattr(doc, "body_text", None) or "").strip()

        # ── Summary length ────────────────────────────────────────────────────
        if len(summary) < min_summary_chars:
            flags.append(
                QualityFlag(
                    doc_id=doc_id,
                    check="summary_too_short",
                    reason=f"summary is {len(summary)} chars (min {min_summary_chars})",
                    severity="error",
                )
            )
        elif len(summary) > max_summary_chars:
            flags.append(
                QualityFlag(
                    doc_id=doc_id,
                    check="summary_too_long",
                    reason=f"summary is {len(summary)} chars (max {max_summary_chars}) — possible body bleed",
                    severity="warning",
                )
            )

        # ── Summary == body bleed ─────────────────────────────────────────────
        if check_summary_bleed and body and summary:
            # If body starts with the summary text verbatim, it's raw body bleed
            # Compare first min(len(summary), 150) chars, normalized
            def _norm(s: str) -> str:
                return " ".join(s.lower().split())

            norm_summary = _norm(summary)
            norm_body = _norm(body)
            cmp_len = min(len(norm_summary), 150)
            if len(norm_summary) >= 50 and norm_body[:cmp_len] == norm_summary[:cmp_len]:
                flags.append(
                    QualityFlag(
                        doc_id=doc_id,
                        check="summary_body_bleed",
                        reason="summary appears to be raw body text (body starts with summary verbatim)",
                        severity="error",
                    )
                )

        # ── AQ count ──────────────────────────────────────────────────────────
        if len(aq) < min_aq_count:
            flags.append(
                QualityFlag(
                    doc_id=doc_id,
                    check="aq_count_low",
                    reason=f"only {len(aq)} AQ strings (min {min_aq_count})",
                    severity="error",
                )
            )
        elif len(aq) > max_aq_count:
            flags.append(
                QualityFlag(
                    doc_id=doc_id,
                    check="aq_count_high",
                    reason=f"{len(aq)} AQ strings (max {max_aq_count})",
                    severity="warning",
                )
            )

        # ── AQ length ─────────────────────────────────────────────────────────
        short_aqs = [q for q in aq if len(q.strip()) < min_aq_length]
        if short_aqs:
            flags.append(
                QualityFlag(
                    doc_id=doc_id,
                    check="aq_too_short",
                    reason=f"{len(short_aqs)} AQ strings are < {min_aq_length} chars",
                    severity="warning",
                )
            )

        # ── Generic AQ patterns ───────────────────────────────────────────────
        if check_generic_aq:
            generic = [q for q in aq if any(p.search(q.strip()) for p in GENERIC_AQ_PATTERNS)]
            if generic:
                flags.append(
                    QualityFlag(
                        doc_id=doc_id,
                        check="generic_aq",
                        reason=f"{len(generic)} generic/placeholder AQ strings: {generic[:2]}",
                        severity="error",
                    )
                )

        # ── Duplicate AQ ─────────────────────────────────────────────────────
        if check_duplicate_aq and aq:
            normalized = [q.strip().lower() for q in aq]
            unique = set(normalized)
            if len(unique) < len(aq):
                dupes = len(aq) - len(unique)
                flags.append(
                    QualityFlag(
                        doc_id=doc_id,
                        check="duplicate_aq",
                        reason=f"{dupes} duplicate AQ strings",
                        severity="warning",
                    )
                )

        return flags

    @classmethod
    def check_many(
        cls,
        docs: list[ContentDocument],
        **kwargs,
    ) -> dict[str, list[QualityFlag]]:
        """
        Check a list of documents.
        Returns {doc_id: [flags]} — only includes docs with at least one flag.
        """
        results: dict[str, list[QualityFlag]] = {}
        for doc in docs:
            flags = cls.check(doc, **kwargs)
            if flags:
                doc_id = str(getattr(doc, "id", None) or "unknown")
                results[doc_id] = flags
        return results
