# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Core abstractions for the enrichment system.

This module defines four things that every other enrichment module depends on:

1. ``EnrichmentBackend`` — the Protocol every backend must satisfy.
2. ``BackendError``      — the exception backends raise on failure.
3. ``ExtractionResult``  — the Pydantic model representing structured output.
4. ``EnrichmentPass``    — the ABC every enrichment pass must implement.

Design philosophy
-----------------
The ``EnrichmentBackend`` Protocol is intentionally minimal: one async method
(``complete``) and one property (``name``).  This means:

- Any async callable that wraps an HTTP request, an agent session, a local
  model, or a test stub is a valid backend with no subclassing required.
- Backends are unaware of Dewie's document model.  They receive a prompt
  string and return a response string.  Parsing and validation are the
  caller's responsibility.
- Swapping backends at runtime (e.g. routing short documents to spaCy and
  long ones to an LLM) requires only changing which backend is selected —
  no changes to the processor or any call site.

``ExtractionResult`` uses ``model_validate`` with lenient defaults so that
partial responses from a backend are accepted rather than rejected.  Missing
fields fall back to empty/null, and the processor logs the gap.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, Field, field_validator

if TYPE_CHECKING:
    from dewie.models.content import ContentDocument
    from dewie.storage.postgres import PostgresClient


class BackendError(Exception):
    """
    Raised by an ``EnrichmentBackend`` implementation when it cannot produce
    a response.

    Common causes:
    - Network timeout or connection refused (HTTP backends).
    - Non-2xx HTTP status code from a remote API.
    - Missing or malformed response field in the API payload.
    - Model loading failure (spaCy backend).

    The ``MetadataProcessor`` catches ``BackendError`` and either retries
    with the fallback backend or marks the document as FAILED.
    """


@runtime_checkable
class EnrichmentBackend(Protocol):
    """
    Minimum interface contract for enrichment backends.

    Any object satisfying this Protocol — regardless of its class hierarchy —
    can be registered in the ``BackendRegistry`` and used by the
    ``EnrichmentRouter``.  No subclassing of any Dewie base class is
    required.

    Implementation targets
    ----------------------
    - **Local NLP** (spaCy, HuggingFace Transformers): Run in-process via
      ``asyncio.to_thread``.  ``complete()`` ignores the prompt string for
      routing but can parse metadata from a fixed header if needed.
    - **Local LLM servers** (Ollama, LM Studio, llama.cpp): HTTP POST to a
      local endpoint.  No API key required.
    - **Remote LLM APIs** (Anthropic, OpenAI, Cohere): HTTP POST with an
      ``Authorization`` header.  ``api_key_env`` config key names the
      environment variable holding the secret.
    - **Agent sessions** (OpenClaw, custom agents): Any async RPC call
      returning a string.  Wrap the call in a thin class implementing this
      Protocol.
    - **Test stubs**: Return a static JSON string without any I/O.

    Contract
    --------
    ``complete`` receives the full extraction prompt (system instruction +
    JSON schema + document content) and **must** return a string that is
    either valid JSON matching ``ExtractionResult`` or is parseable into it
    with lenient validation.

    ``complete`` **must** raise ``BackendError`` (not a generic exception)
    when it cannot produce a response, so the processor can apply fallback
    logic cleanly.
    """

    async def complete(self, prompt: str) -> str:
        """
        Send the extraction prompt to the backend and return raw response text.

        Args:
            prompt: Full extraction prompt including system instruction, JSON
                    schema contract, and document content.  Built by
                    ``enrichment.schema.build_extraction_prompt``.

        Returns:
            Raw text response from the backend.  Should be a JSON object
            string matching ``ExtractionResult``.  The caller is responsible
            for parsing and validating the response.

        Raises:
            BackendError: On any failure that prevents returning a response.
                          Do not raise bare ``Exception`` — the processor
                          uses ``BackendError`` to distinguish backend failures
                          from programming errors.
        """
        ...

    @property
    def name(self) -> str:
        """
        Human-readable identifier for this backend.

        Used for:
        - Registry lookup (``BackendRegistry.get(name)``).
        - Log messages and error reporting.
        - Config-driven routing rules (``use_backend: <name>``).

        Must be unique within a ``BackendRegistry``.
        """
        ...


# ── ExtractionResult ─────────────────────────────────────────────────────────


class EntityExtraction(BaseModel):
    """A single named entity returned by the extraction backend."""

    text: str = Field(description="Surface form of the entity.")
    label: str = Field(
        default="UNKNOWN",
        description="Entity type label (e.g. ORG, PERSON, GPE).",
    )


class ExtractionResult(BaseModel):
    """
    Structured extraction output returned by every enrichment backend.

    This is the JSON contract between backends and the ``MetadataProcessor``.
    All fields are optional with safe defaults so that partial responses from
    a backend are accepted and logged rather than rejected entirely.

    Validators clamp numeric fields to their valid ranges so out-of-range
    values from an LLM do not cause downstream errors.

    Usage
    -----
    ::

        raw_json = await backend.complete(prompt)
        result = ExtractionResult.model_validate_json(raw_json)
        # or, if the backend wraps the JSON in prose:
        result = ExtractionResult.model_validate(extract_json_object(raw_json))
    """

    document_type: str | None = Field(
        default=None,
        description=(
            "Coarse format classification.  Should match a ``DocumentType`` "
            "enum value (blog_post, tweet, academic_paper, …)."
        ),
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="High-signal token lemmas ranked by relevance.",
    )
    themes: list[str] = Field(
        default_factory=list,
        description="Higher-level multi-word thematic concepts.",
    )
    entities: list[EntityExtraction] = Field(
        default_factory=list,
        description="Named entities extracted from the document.",
    )
    summary: str = Field(
        default="",
        description=(
            "Concise summary of the document, ≤250 tokens (~1–2 sentences).  "
            "Designed to give agents enough signal to decide whether to explore "
            "this node further."
        ),
    )
    enrichment_quality_score: int | None = Field(
        default=None,
        description="Document quality estimate, 0–100.",
    )
    sentiment: float | None = Field(
        default=None,
        description="Polarity score: -1.0 (very negative) to +1.0 (very positive).",
    )
    tone: str | None = Field(
        default=None,
        description=(
            "Qualitative tone descriptor.  Examples: optimistic, critical, "
            "neutral, informative, satirical, academic."
        ),
    )
    author: str | None = Field(
        default=None,
        description="Author name(s) if clearly stated in the document, otherwise None.",
    )
    reading_level: str | None = Field(
        default=None,
        description=(
            "Estimated reading level.  One of: quick_read, standard, "
            "long_read, deep_dive, academic."
        ),
    )
    language: str = Field(
        default="en",
        description="ISO 639-1 language code of the source document.",
    )
    answers_questions: list[str] = Field(
        default_factory=list,
        description=(
            "4-8 questions this document directly answers. Written from an agent's "
            "perspective: 'What is...', 'How does...', 'Why did...'. "
            "Each question MUST be self-contained and searchable on its own: name "
            "the subject and the specific terms explicitly (e.g. 'What term did "
            "Friedrich Bauer coin?' with the answer 'software engineering' also "
            "named where possible — never 'this person', 'it', or 'the document'). "
            "These strings are matched against user queries by full-text and "
            "vector search; pronouns and vague references make them unfindable. "
            "Helps agents decide if this node answers their current query."
        ),
    )
    missing_coverage: list[str] = Field(
        default_factory=list,
        description=(
            "1-3 related aspects or angles NOT covered by this document. "
            "Helps agents decide to keep exploring rather than stopping here."
        ),
    )
    embed_summary: str = Field(
        default="",
        description=(
            "Retrieval-dense summary for vector embedding. 200-300 words of dense "
            "informational prose covering key facts, named entities, specific claims, "
            "data points, and conclusions. Not a description of the document."
        ),
    )
    alternate_terms: list[str] = Field(
        default_factory=list,
        description=(
            "Synonyms, acronym expansions, and alternate names for key entities in this "
            "document. Used for query expansion at search time. E.g. ['NBA', 'basketball', "
            "'National Basketball Association']. Only include genuinely equivalent or "
            "strongly associated terms — do not speculate."
        ),
    )

    @field_validator("enrichment_quality_score", mode="before")
    @classmethod
    def clamp_enrichment_quality_score(cls, v: object) -> int | None:
        """Clamp enrichment_quality_score to [0, 100]; discard non-numeric values."""
        if v is None:
            return None
        try:
            return max(0, min(100, int(v)))
        except (TypeError, ValueError):
            return None

    @field_validator("sentiment", mode="before")
    @classmethod
    def clamp_sentiment(cls, v: object) -> float | None:
        """Clamp sentiment to [-1.0, 1.0]; discard non-numeric values."""
        if v is None:
            return None
        try:
            return max(-1.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return None

    @field_validator("entities", mode="before")
    @classmethod
    def normalise_entities(cls, v: object) -> list[dict]:  # type: ignore[type-arg]
        """
        Accept entities as either dicts or plain strings. Caps at 10 entries.

        Some backends return entities as a list of strings rather than objects.
        Plain strings are promoted to ``{"text": s, "label": "UNKNOWN"}``.
        Large entity lists are the most common cause of mid-stream JSON truncation
        (long arrays push response size over the streaming threshold).
        """
        if not isinstance(v, list):
            return []
        result = []
        for item in v:
            if isinstance(item, str):
                result.append({"text": item, "label": "UNKNOWN"})
            elif isinstance(item, dict):
                result.append(item)
        return result[:10]

    @field_validator("answers_questions", "keywords", mode="before")
    @classmethod
    def coerce_to_str_list(cls, v: object) -> list[str]:
        """
        Coerce non-list values (false, null, bare string) to a list.

        Some LLM responses return ``false`` or ``null`` for list fields.
        A bare string is wrapped in a single-element list.
        """
        if isinstance(v, list):
            return [str(item) for item in v if item is not None]
        if isinstance(v, str):
            return [v]
        return []


# ── EnrichmentPass ABC ─────────────────────────────────────────────────────────


class EnrichmentPass(ABC):
    """
    Abstract base class for enrichment pipeline passes.

    Each pass performs one step of the enrichment pipeline (e.g. metadata
    extraction, chunking, embedding).  Passes are registered in a
    ``PassRegistry`` at startup and executed in registration order.

    This ABC provides the plugin boundary that allows external packages
    (cloud layer, community plugins) to register additional enrichment
    passes without modifying Dewie's core code.

    Usage
    -----
    ::

        class MyCustomPass(EnrichmentPass):
            name = "my_custom"

            async def run(self, doc: ContentDocument, pg: PostgresClient) -> None:
                # Perform custom enrichment logic
                ...

        registry = PassRegistry()
        registry.register(MyCustomPass())

    Subclasses must set a unique ``name`` and implement ``run()``.
    """

    name: str = ""

    @abstractmethod
    async def run(self, doc: ContentDocument, pg: PostgresClient) -> None:
        """
        Execute this enrichment pass on the given document.

        The pass should mutate ``doc`` in-place and use ``pg`` to persist
        any results.  The method may also update the document status.

        Args:
            doc: Document to enrich.  Fields are mutated in-place.
            pg:  PostgreSQL client for persisting enrichment results.

        Raises:
            Exception: Any exception is caught by the pipeline orchestrator
                       and logged; the pass is considered failed.
        """
        ...
