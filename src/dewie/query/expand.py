# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Query expansion with lightweight synonym tables.

Appends a few synonyms for common query words so hybrid search has more
terms to match on, without changing the original intent. Pure string
enrichment — no model call needed.

Domain-specific expansion (e.g. industry jargon, product taxonomies) is
intentionally left to operators: extend ``_SYNONYMS`` for your corpus.
"""

from __future__ import annotations

# ── Synonym pairs for common query words (bidirectional, corpus-agnostic) ──────
_SYNONYMS: dict[str, list[str]] = {
    "ask": ["question", "query"],
    "search": ["find", "look up"],
    "find": ["search", "locate"],
    "company": ["corporation", "firm", "organization"],
    "report": ["filing", "statement", "document"],
    "document": ["doc", "file", "record"],
    "guide": ["tutorial", "walkthrough", "how-to"],
    "error": ["failure", "exception", "bug"],
    "config": ["configuration", "settings", "setup"],
    "install": ["installation", "setup"],
}


def _expand_synonyms(query: str) -> str:
    """Append synonyms for any recognized query words."""
    q_lower = query.lower()
    additions: list[str] = []

    for term, expansions in _SYNONYMS.items():
        if term in q_lower:
            additions.extend(expansions)

    # Deduplicate while preserving order; skip terms already in the query
    seen: set[str] = set()
    unique: list[str] = []
    for a in additions:
        a_lower = a.lower()
        if a_lower in seen or a_lower in q_lower:
            continue
        seen.add(a_lower)
        unique.append(a)

    if not unique:
        return query

    return f"{query} {' '.join(unique)}"


def expand_query(query: str) -> str:
    """
    Expand a search query by appending synonyms for common words.

    The expanded query is the original query + appended synonym terms,
    giving hybrid search more terms to match on without changing intent.

    Examples:
        >>> expand_query("install guide")
        'install guide installation setup tutorial walkthrough how-to'

        >>> expand_query("how to bake bread")
        'how to bake bread'
    """
    if not query or not query.strip():
        return query

    return _expand_synonyms(query.strip())
