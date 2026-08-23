# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Extraction prompt builder for LLM-based enrichment backends.

The prompt is the contract between the system and any text-completion backend.
It specifies:

1. A system instruction explaining the task and the expected output format.
2. A JSON schema definition with field types, constraints, and examples.
3. The document content (title + body), capped at ``MAX_EXTRACTION_CHARS``.
4. An explicit closing instruction to return only raw JSON with no prose.

Design decisions
----------------
- The prompt is a plain string, not a chat message array.  This keeps it
  compatible with completion-mode APIs (Ollama ``/api/generate``) and
  chat-mode APIs (OpenAI ``/v1/chat/completions``) without adaptation.
  ``HttpBackend`` wraps it in the appropriate structure for each API shape.

- The schema description is embedded in the prompt rather than passed as a
  JSON Schema object.  This works across backends that do not support
  structured output modes.

- ``MAX_EXTRACTION_CHARS`` caps the body length sent to the backend.  Long
  documents are truncated with a note appended so the backend is aware.
  The full body is still used for relationship building — only the backend
  input is capped.

- ``MAX_SUMMARY_CHARS`` is the hard cap applied to the summary *after*
  the backend returns it.  This prevents oversized summaries from being
  persisted even if the backend ignores the token budget instruction.
"""

from __future__ import annotations

import os

MAX_EXTRACTION_CHARS: int = int(os.environ.get("DEWIE_MAX_EXTRACTION_CHARS", "80000"))
"""
Maximum number of characters from the document body sent to the enrichment
backend.  Corresponds to roughly 20,000 tokens for typical English text,
which comfortably fits within the context window of most 3B+ parameter models.
"""

MAX_SUMMARY_CHARS: int = 1_500
"""
Hard cap applied to the ``summary`` field after the backend returns it.
~1,500 characters ≈ 250–375 tokens depending on tokeniser.
"""

MAX_EMBED_SUMMARY_CHARS: int = 25_000
"""
Hard cap applied to the ``embed_summary`` field after the backend returns it.
~25,000 characters ≈ 6,250 tokens — uses ~75% of text-embedding-3-small's
8,191 token context window, leaving headroom for title + AQ questions that
are prepended at embedding time.
"""

_SCHEMA_INSTRUCTION = """\
You are a document metadata extraction assistant.
Analyse the document provided and return ONLY a valid JSON object.
Do not include any prose, markdown, or explanation outside the JSON.

The JSON object must have the following fields:

{
  "document_type": string,      // One of: news_article, blog_post, academic_paper,
                                //   forum_post, social_media, documentation,
                                //   video, podcast, other
  "author":        string|null, // Author name if clearly stated in the document,
                                //   otherwise null
  "tone":          string,      // One of: optimistic, critical, neutral,
                                //   informative, satirical, technical, emotional
  "reading_level": string,      // One of: quick_read (under 5 min),
                                //   standard (5-15 min), long_read (15-30 min),
                                //   deep_dive (30+ min),
                                //   academic (formal dense academic writing)
  "keywords":     [string],     // Top 10–15 high-signal token lemmas, ranked
                                //   by relevance descending
  "themes":       [string],     // 3–8 higher-level multi-word thematic concepts
                                //   (e.g. "knowledge graphs", "supply chain risk")
  "entities":     [             // Up to 10 most important named entities only.
    { "text": string, "label": string }
                                // label: ORG, PERSON, GPE, PRODUCT, EVENT,
                                //   WORK_OF_ART, LAW, or OTHER
  ],
  "summary":      string,       // Concise summary in 1–2 sentences, max 250 tokens.
                                //   Must be intriguing enough to spark further
                                //   exploration of the topic.
  "embed_summary":     string,  // Retrieval-dense summary for vector embedding.
                                //   Scale length to document depth:
                                //   Short content (<500 words): 200-400 words.
                                //   Medium (500-2000 words): 400-1000 words.
                                //   Long/dense (2000+ words, papers, legal, tech): 1000-5000 words.
                                //   MUST include: key facts, named entities, specific claims,
                                //   data points, statistics, quotes, methodology, conclusions,
                                //   and any novel contributions or findings.
                                //   Do NOT include meta-commentary about the document.
                                //   Write as dense informational prose, not as a description
                                //   of what the document discusses. Preserve technical
                                //   terminology exactly as used in the source.
  "enrichment_quality_score": integer,     // Informational density score, 0–100.
                                //   Score based on BODY LENGTH + CONTENT DEPTH together.
                                //   Be ruthless — most web content is mediocre.
                                //   Anchors:
                                //     0–15:  stub, near-empty, just a headline or blurb
                                //     16–35: very thin (<500 chars or <100 words), little substance
                                //     36–55: short-form, limited depth, mostly surface-level
                                //     56–70: decent — substantive but not comprehensive
                                //     71–84: strong — clear structure, named entities, specific facts/data
                                //     85–94: excellent — original analysis, primary data, comprehensive
                                //     95–100: exceptional — rare, must have deep original reporting or research
                                //   A 1–2 sentence blog note MUST score below 40.
                                //   A tweet-length item MUST score below 20.
                                //   A full news article (500+ words) starts at 55 minimum.
                                //   Do NOT cluster scores in the 80–90 range.
  "sentiment":    float,        // Polarity: -1.0 (very negative) to +1.0
                                //   (very positive)
  "language":     string,       // ISO 639-1 code (e.g. "en", "fr", "de")
  "answers_questions": [string], // 4-8 questions this document directly answers,
                                //   written from an agent's perspective. E.g.:
                                //   "What is Jaccard similarity?"
                                //   "How do graph databases handle traversal?"
  "missing_coverage":  [string] // 1-3 related aspects NOT covered by this document. E.g.:
                                //   "Does not cover implementation in Python"
                                //   "No discussion of performance at scale"
  "alternate_terms":   [string] // Synonyms, acronym expansions, and alternate names for
                                //   key entities in this document. Used for query expansion.
                                //   E.g. ["NBA", "basketball", "National Basketball Association"]
                                //   ["POTUS", "president", "president of the United States"]
                                //   ["AI", "artificial intelligence", "machine learning"]
                                //   Include 5-15 terms. Only genuinely equivalent or strongly
                                //   associated terms — do not speculate or add tangential concepts.
}

Rules:
- Return ONLY the JSON object. No markdown fences. No prose before or after.
- All list fields may be empty arrays [] if no values apply.
- Null fields are not permitted except for "author" which may be null.
- answers_questions and missing_coverage may be empty arrays [] if not applicable.
"""


def build_extraction_prompt(title: str, body: str) -> str:
    """
    Build the full extraction prompt for a document.

    The prompt is a single string suitable for both completion-mode and
    chat-mode LLM APIs.  The caller (``HttpBackend``) wraps it in the
    appropriate API payload structure.

    Args:
        title: Document title.  May be an empty string for untitled content.
        body:  Full document body text.  Truncated to ``MAX_EXTRACTION_CHARS``
               if longer; a note is appended to inform the backend.

    Returns:
        A prompt string containing the system instruction, JSON schema
        definition, and document content.

    Notes:
        This function is a pure transformation — no I/O, no side effects.
        It is safe to call from any context.

    Example::

        prompt = build_extraction_prompt(
            title="Graph Databases Explained",
            body="Neo4j is a property graph database ...",
        )
        raw_json = await backend.complete(prompt)
    """
    truncated = False
    if len(body) > MAX_EXTRACTION_CHARS:
        body = body[:MAX_EXTRACTION_CHARS]
        truncated = True

    document_section = f"TITLE: {title}\n\nBODY:\n{body}"
    if truncated:
        document_section += (
            "\n\n[NOTE: Document body was truncated to fit the extraction window. "
            "Base the summary and analysis on the content above.]"
        )

    return f"{_SCHEMA_INSTRUCTION}\n\n---\nDOCUMENT:\n{document_section}\n---\n\nJSON:"
