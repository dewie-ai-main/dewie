# SPDX-License-Identifier: FSL-1.1-ALv2
# Copyright 2026 Alexander Atkins
# Licensed under the Functional Source License, Version 1.1, ALv2 Future License.
# Each version becomes Apache-2.0 two years after its release. See LICENSE.

"""
Pydantic models representing a content document as it flows through the system.

Lifecycle
---------
1. PENDING  — stub created at ingest time; only URL/title/source are reliable.
2. PROCESSING — enrichment backend has claimed the document.
3. READY    — all enrichment fields populated; document is queryable.
4. FAILED   — enrichment failed after all retries; operator intervention needed.

Field population
----------------
Fields marked "ingest" are available immediately after ingestion.
Fields marked "enrichment" are populated by the enrichment pipeline and may be
absent (default values) until the document reaches READY status.
"""

import uuid
from datetime import datetime
from enum import StrEnum
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, HttpUrl

# UUID5 namespace for stable document IDs derived from URLs
_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # NAMESPACE_URL


def _normalize_url(url: str) -> str:
    """Strip utm_* params, trailing slash, www. prefix for stable hashing."""
    parsed = urlparse(url.strip().rstrip("/"))
    host = parsed.netloc.lstrip("www.")
    params = [(k, v) for k, v in parse_qsl(parsed.query) if not k.startswith("utm_")]
    clean = parsed._replace(netloc=host, query=urlencode(params))
    return urlunparse(clean)


def make_doc_id(url: str) -> uuid.UUID:
    """Return a deterministic UUID5 for the given URL."""
    return uuid.uuid5(_NAMESPACE, _normalize_url(url))


class Scope(StrEnum):
    """
    Retrieval scope — determines which users can see a workspace or corpus.

    PUBLIC
        Accessible to any valid API key or authenticated user.

    INTERNAL_ONLY
        Accessible only to users within the same workspace.

    PRIVATE
        Accessible only to the owner of the workspace/corpus.
    """

    PUBLIC = "public"
    INTERNAL_ONLY = "internal_only"
    PRIVATE = "private"


class ContentStatus(StrEnum):
    """Processing status of a content document."""

    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    TERMINAL = "terminal"  # permanent failure — no retry, moved to review_queue


class DocumentType(StrEnum):
    """
    Coarse classification of document format, assigned during enrichment.

    Used by the enrichment router (to select backends), by the browse API
    (to filter neighbor previews), and by agents (to interpret content).
    """

    BLOG_POST = "blog_post"
    TWEET = "tweet"
    ACADEMIC_PAPER = "academic_paper"
    NEWS_ARTICLE = "news_article"
    FORUM_POST = "forum_post"
    SOCIAL_MEDIA = "social_media"
    DOCUMENTATION = "documentation"
    VIDEO = "video"
    VIDEO_TRANSCRIPT = "video_transcript"
    AUDIO_TRANSCRIPT = "audio_transcript"
    PODCAST = "podcast"
    OTHER = "other"


class ReadingLevel(StrEnum):
    """Estimated reading level/length category assigned by enrichment."""

    QUICK_READ = "quick_read"  # < 5 min
    STANDARD = "standard"  # 5-15 min
    LONG_READ = "long_read"  # 15-30 min
    DEEP_DIVE = "deep_dive"  # 30+ min
    ACADEMIC = "academic"  # formal/dense regardless of length


class ContentDocument(BaseModel):
    """
    A single unit of ingested content with its enriched metadata attached.

    This model is the central data structure in Dewie.  It is passed
    through the ingest → enrich → store → query pipeline.  Not all fields
    are populated at every stage — see the field-level ``description``
    annotations for when each field becomes available.

    ``body`` is intentionally excluded from serialisation (``exclude=True``).
    It is held in memory during enrichment only; the persisted representation
    uses ``summary`` instead.
    """

    # ── Identity ──────────────────────────────────────────────────────────────

    id: UUID = Field(
        default_factory=uuid4,
        description="Immutable primary key assigned at ingest time.",
    )
    url: str = Field(
        description="Canonical URL of the content.  Unique across the corpus.",
    )
    instance_id: str | None = Field(
        default=None,
        description=(
            "UUID of the Dewie instance that ingested this document. "
            "Used for deduplication across federated nodes. "
            "Populated automatically on upsert if None."
        ),
    )
    title: str = Field(
        default="",
        description="Page or article title.  Set at ingest time.",
    )
    body: str = Field(
        default="",
        exclude=True,
        description=(
            "Full document text.  Held in memory during enrichment only; "
            "not stored or serialised.  Use ``summary`` for the persisted excerpt."
        ),
    )
    store_body: bool = Field(
        default=True,
        exclude=True,
        description=(
            "If True, cache the full body in Redis after enrichment. "
            "Set False for public/licensed content (news, paywalled sources) "
            "where we store metadata+URL only — the URL is the product. "
            "Body is always available in memory during the enrichment window."
        ),
    )
    summary: str = Field(
        default="",
        description=(
            "LLM-generated summary capped at ~250 tokens (~1–2 sentences). "
            "Populated by the enrichment pipeline.  This is the primary signal "
            "agents use to decide whether to explore a node further."
        ),
    )
    embed_summary: str = Field(
        default="",
        description="Retrieval-dense summary for embedding (200-300 words, LLM-generated).",
    )
    source: str = Field(
        default="",
        description="Origin feed label or domain (e.g. 'example.com').",
    )
    ingested_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="UTC timestamp of first ingestion.",
    )
    status: ContentStatus = Field(
        default=ContentStatus.PENDING,
        description="Current processing status.",
    )

    # ── Crawler provenance ────────────────────────────────────────────────────

    crawl_session: UUID | None = Field(
        default=None,
        description="UUID of the crawl session that produced this document, if any.",
    )

    # ── Enrichment fields ─────────────────────────────────────────────────────
    # All fields below are populated by the enrichment pipeline.
    # They carry default values so that PENDING documents are valid models.

    document_type: DocumentType | None = Field(
        default=None,
        description=(
            "Coarse format classification assigned by the enrichment backend.  "
            "Drives routing decisions and is exposed in browse preview cards."
        ),
    )
    topics: list[str] = Field(
        default_factory=list,
        description=(
            "Coarse single-word topic labels derived from noun-chunk frequency "
            "or LLM classification.  Used as primary graph relationship keys."
        ),
    )
    themes: list[str] = Field(
        default_factory=list,
        description=(
            "Higher-level, multi-word thematic concepts (e.g. 'knowledge graphs', "
            "'distributed systems').  Distinct from topics; intended as a human- "
            "and agent-readable summary of what the document is about."
        ),
    )
    keywords: list[str] = Field(
        default_factory=list,
        description="High-signal token lemmas ranked by TF-IDF or backend scoring.",
    )
    entities: list[str] = Field(
        default_factory=list,
        description="Named entities (ORG, PERSON, GPE, …) extracted from the text.",
    )
    sentiment: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
        description="Polarity score: -1.0 (negative) to +1.0 (positive).",
    )
    enrichment_quality_score: int | None = Field(
        default=None,
        ge=0,
        le=100,
        description=(
            "Backend-assigned quality estimate (0–100).  Higher values indicate "
            "richer content, clearer structure, and higher relevance signal."
        ),
    )
    tone: str | None = Field(
        default=None,
        description=(
            "Qualitative tone descriptor assigned by the enrichment backend.  "
            "Examples: 'optimistic', 'critical', 'neutral', 'informative', "
            "'satirical'.  Helps agents filter content by style."
        ),
    )
    author: str | None = Field(
        default=None,
        description="Author name(s) extracted by enrichment backend.",
    )
    reading_level: ReadingLevel | None = Field(
        default=None,
        description="Estimated reading level/length category assigned by enrichment.",
    )
    language: str = Field(
        default="en", description="ISO 639-1 language code detected during enrichment."
    )
    answers_questions: list[str] = Field(
        default_factory=list,
        description=(
            "Questions this document directly answers. Populated by enrichment. "
            "Helps agents determine if this node satisfies their current information need."
        ),
    )
    missing_coverage: list[str] = Field(
        default_factory=list,
        description=(
            "Related aspects NOT covered by this document. Populated by enrichment. "
            "Helps agents decide to keep exploring rather than stopping here."
        ),
    )
    alternate_terms: list[str] = Field(
        default_factory=list,
        description=(
            "Synonyms, acronym expansions, and alternate names for key entities in this "
            "document. Used for query expansion at search time."
        ),
    )

    # ── Enrichment versioning ─────────────────────────────────────────────────

    enrichment_version: int = Field(
        default=0,
        description=(
            "Incremented each time the document is re-enriched.  "
            "Bump the target version to force re-enrichment when prompts or "
            "models change."
        ),
    )
    embedding_model: str | None = Field(
        default=None,
        description="Name of the embedding model used to produce the vector.",
    )
    enriched_at: datetime | None = Field(
        default=None,
        description="UTC timestamp of the most recent successful enrichment run.",
    )

    # ── Publication date ──────────────────────────────────────────────────────

    published_at: datetime | None = Field(
        default=None,
        description=(
            "UTC timestamp when the original article/document was published. "
            "Extracted from OG meta (article:published_time), JSON-LD (datePublished), "
            "or RSS feed entry date at ingest time. "
            "Backfilled via URL-pattern parsing for existing corpus. "
            "None if publication date could not be determined."
        ),
    )

    # ── Corpus tagging ────────────────────────────────────────────────────────

    corpus_id: str | None = Field(
        default=None,
        description=(
            "Opaque corpus identifier for grouping documents by dataset origin. "
            "Convention: 'beir:DATASET', 'customer:NAME', 'dogfood:SOURCE'. "
            "NULL for organically-ingested documents."
        ),
    )

    # ── Gap-fill tagging ──────────────────────────────────────────────────────

    gap_fill: bool = Field(
        default=False,
        description=(
            "True if this document was ingested by the search-queue gap-fill worker "
            "(Brave fallback). Never reset to False once set."
        ),
    )

    # ── Paywall detection (set during ingest) ─────────────────────────────────

    paywall_detected: bool = Field(
        default=False,
        description="True if a paywall was detected when fetching the full article.",
    )
    paywall_type: str = Field(
        default="none",
        description=(
            "Paywall type detected during ingest. "
            "One of: none | soft | hard | metered. "
            "soft = subscribe-wall text present; hard = body too short after extraction; "
            "metered = Schema.org isAccessibleForFree:false or known paywall SDK detected."
        ),
    )

    # ── Terminal / skip tracking ──────────────────────────────────────────────

    skip_reason: str = Field(
        default="",
        description=(
            "Reason this document was skipped or marked terminal. "
            "Examples: 'paywall_no_body', 'stub', 'blocked_source'. "
            "Empty string means the document is not skipped."
        ),
    )

    # ── Visibility ────────────────────────────────────────────────────────────

    visibility: str = Field(
        default="public",
        description="public | private. Private docs only appear in workspace-scoped searches.",
    )

    # ── Location (geo-expansion stub) ─────────────────────────────────────────

    location: dict | None = Field(
        default=None,
        description=(
            "Location data stored as JSONB. Currently a stub — accepted but not "
            "used for search filtering. Future geo-expansion will use lat/lon/radius_km "
            "for location-based ranking."
        ),
    )

    # ── Ownership ─────────────────────────────────────────────────────────────

    owner_user_id: str | None = Field(
        default=None,
        description=(
            "UUID of the user who ingested this document. "
            "NULL for system-ingested documents. "
            "Used to scope delete and private-visibility operations."
        ),
    )

    model_config = {"from_attributes": True}

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_url(cls, url: str, **kwargs) -> "ContentDocument":
        """Create a ContentDocument with a deterministic ID derived from the URL."""
        return cls(id=make_doc_id(url), url=url, **kwargs)

    def to_export_dict(self) -> dict:
        """Return a dictionary of fields safe for export.

        Excludes answers_questions (forbidden) and internal fields.
        Dates are serialized as ISO 8601 strings.
        """
        return {
            "id": str(self.id),
            "url": self.url,
            "title": self.title,
            "summary": self.summary,
            "topics": list(self.topics) if self.topics else [],
            "keywords": list(self.keywords) if self.keywords else [],
            "entities": list(self.entities) if self.entities else [],
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "enriched_at": self.enriched_at.isoformat() if self.enriched_at else None,
            "corpus_id": self.corpus_id,
            "tags": [],
        }


class IngestRequest(BaseModel):
    """Payload for submitting a URL or RSS feed for ingestion."""

    url: HttpUrl = Field(
        description="URL to ingest (article page or RSS feed endpoint).",
    )
    source_label: str = Field(
        default="",
        description="Optional human-readable source name overriding domain detection.",
    )
    title: str | None = Field(
        default=None,
        description="Optional pre-fetched title. When provided alongside body, skips re-fetching the URL.",
    )
    body: str | None = Field(
        default=None,
        description="Optional pre-fetched body text. When provided, the /ingest endpoint uses it directly instead of re-fetching the URL.",
    )
    min_body_chars: int = Field(
        default=-1,
        description=(
            "Minimum body length in characters for this document to be accepted. "
            "-1 disables the check. Set per feed group via dewie.yml ingest.quality_filter "
            "or ingest.feed_groups[].quality_filter.min_body_chars."
        ),
    )
    corpus_id: str | None = Field(
        default=None,
        description=(
            "Opaque corpus identifier forwarded to the document record. "
            "Convention: 'beir:DATASET', 'customer:NAME', 'dogfood:SOURCE'."
        ),
    )
    gap_fill: bool = Field(
        default=False,
        description="When True, tag the ingested document(s) as gap-fill (Brave fallback).",
    )
    visibility: str = Field(
        default="public",
        description="public | private. Private docs only appear in searches for the owning tenant.",
    )
    enrichment_provider: str | None = Field(
        default=None,
        description=(
            "Optional enrichment provider override for this ingest request. "
            "Must be provided together with enrichment_model."
        ),
    )
    enrichment_model: str | None = Field(
        default=None,
        description=(
            "Optional enrichment model override for this ingest request. "
            "Must be provided together with enrichment_provider."
        ),
    )
