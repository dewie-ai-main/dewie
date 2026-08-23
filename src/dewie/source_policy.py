# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
source_policy.py — Per-source storage and enrichment policy.

Every source that enters Dewie has a policy:
  - store_body: bool — cache full body in Redis after enrichment
  - source_type: 'owned' | 'licensed' | 'public'
  - rank_tier: int — used for result ranking (1=primary docs, 2=analysis, 3=news)

Policy rules:
  owned    → store_body=True  (you own the corpus — private docs, dev corpora)
  licensed → store_body=False (you don't own it — news, financial data)
  public   → store_body=False by default, True if explicitly opted in

This is intentionally a flat registry, not config-file-driven, because
policy decisions need to be deliberate and code-reviewed.

Usage:
    from dewie.source_policy import get_policy, SOURCE_POLICIES
    policy = get_policy("reuters.com")
    if policy.store_body:
        cache content...
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SourcePolicy:
    store_body: bool
    source_type: str  # 'owned' | 'licensed' | 'public'
    rank_tier: int  # 1=primary docs, 2=analysis/research, 3=news/feeds
    description: str = ""


# ── Source registry ───────────────────────────────────────────────────────────
#
# Tier 1 — Primary documentation (owned/operator corpus)
#   Full body stored.
#
# Tier 2 — Research / analysis / practitioner blogs
#   Full body stored.
#
# Tier 3 — News / licensed content
#   Metadata + URL only. No body stored.
#

SOURCE_POLICIES: dict[str, SourcePolicy] = {
    # ── Tier 1: Owned / operator corpora ─────────────────────────────────────
    "dewie-docs": SourcePolicy(True, "owned", 1, "Dewie documentation"),
    "docker-docs": SourcePolicy(True, "owned", 1, "Docker documentation"),
    "wikipedia": SourcePolicy(True, "public", 1, "Wikipedia (CC BY-SA, store OK)"),
    "en.wikipedia.org": SourcePolicy(True, "public", 1, "Wikipedia"),
    # ── Tier 2: Research / practitioner blogs ────────────────────────────────
    "thegradient.pub": SourcePolicy(True, "public", 2, "The Gradient"),
    "bair.berkeley.edu": SourcePolicy(True, "public", 2, "BAIR Blog"),
    "lilianweng.github.io": SourcePolicy(True, "public", 2, "Lilian Weng"),
    "huggingface.co": SourcePolicy(True, "public", 2, "HuggingFace Blog"),
    "importai.substack.com": SourcePolicy(True, "public", 2, "Import AI"),
    "github.blog": SourcePolicy(True, "public", 2, "GitHub Blog"),
    "stackoverflow.blog": SourcePolicy(True, "public", 2, "Stack Overflow Blog"),
    "marginalrevolution.com": SourcePolicy(True, "public", 2, "Marginal Revolution"),
    "www.econlib.org": SourcePolicy(True, "public", 2, "EconLib"),
    # ── Tier 3: News / licensed — metadata + URL only ────────────────────────
    # A few illustrative licensed news sources. Operators extend this registry
    # for their own corpus; these are examples, not an exhaustive policy set.
    "www.bbc.com": SourcePolicy(False, "licensed", 3, "BBC News"),
    "www.theguardian.com": SourcePolicy(False, "licensed", 3, "The Guardian"),
    "reuters.com": SourcePolicy(False, "licensed", 3, "Reuters"),
    "feeds.reuters.com": SourcePolicy(False, "licensed", 3, "Reuters"),
    # Tech news (licensed)
    "arstechnica.com": SourcePolicy(False, "licensed", 3, "Ars Technica"),
    "www.technologyreview.com": SourcePolicy(False, "licensed", 3, "MIT Tech Review"),
    "techcrunch.com": SourcePolicy(False, "licensed", 3, "TechCrunch"),
    # Aggregators
    "news.ycombinator.com": SourcePolicy(False, "public", 3, "Hacker News"),
}

_DEFAULT_PUBLIC = SourcePolicy(True, "public", 2, "default public")
_DEFAULT_NEWS = SourcePolicy(False, "licensed", 3, "default news/licensed")


def get_policy(source: str) -> SourcePolicy:
    """
    Return the storage/rank policy for a source label or domain.
    Falls back to a sensible default if source not explicitly registered.
    """
    if source in SOURCE_POLICIES:
        return SOURCE_POLICIES[source]

    # Heuristic for unregistered sources: news-ish domains get no-body
    news_tlds = (
        ".com/news",
        "news.",
        "reuters",
        "bloomberg",
        "cnbc",
        "bbc",
        "guardian",
        "nytimes",
        "wsj",
        "ft.com",
        "politico",
    )
    if any(hint in source.lower() for hint in news_tlds):
        return _DEFAULT_NEWS

    return _DEFAULT_PUBLIC


def should_store_body(source: str) -> bool:
    return get_policy(source).store_body


def rank_tier(source: str) -> int:
    return get_policy(source).rank_tier
