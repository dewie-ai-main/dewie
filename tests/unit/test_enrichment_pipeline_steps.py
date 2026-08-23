"""
Per-step unit tests for the enrichment pipeline.

Each test exercises exactly one pipeline step with mocked dependencies.
All external I/O (DB, HTTP, Redis) is mocked; no live services required.
"""

from __future__ import annotations

import json
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dewie.enrichment.backends.passthrough import PassthroughBackend
from dewie.enrichment.base import ExtractionResult
from dewie.enrichment.processor import (
    MetadataProcessor,
    _apply_result_to_doc,
    _parse_extraction_result,
)
from dewie.models.content import (
    ContentDocument,
    ContentStatus,
    DocumentType,
    ReadingLevel,
)
from dewie.pipeline import build_embed_text
from dewie.storage.body_store import load_body, save_body


@pytest.fixture(autouse=True)
def _force_single_pass(monkeypatch):
    """Pin single_pass mode: a local dewie.yml with enrichment_mode: dual_pass
    would make these tests call a live LLM, overriding the mocked backend."""
    from dewie.config import settings

    monkeypatch.setattr(settings, "enrichment_mode", "single_pass")


# ── Helpers ────────────────────────────────────────────────────────────────────

VALID_EXTRACTION_JSON = json.dumps(
    {
        "document_type": "blog_post",
        "author": "Alice Smith",
        "tone": "informative",
        "reading_level": "standard",
        "keywords": ["ai", "graph", "traversal"],
        "themes": ["knowledge graphs", "ai systems"],
        "entities": [{"text": "OpenAI", "label": "ORG"}],
        "summary": "A guide to AI and graph traversal systems.",
        "enrichment_quality_score": 75,
        "sentiment": 0.4,
        "language": "en",
        "answers_questions": ["What is graph traversal?"],
        "missing_coverage": ["Production deployment"],
    }
)


def _make_doc(**kwargs) -> ContentDocument:
    defaults = {
        "url": "https://example.com/test",
        "title": "Test Document",
        "source": "example.com",
    }
    defaults.update(kwargs)
    doc = ContentDocument(**defaults)
    doc.body = kwargs.get("body", "This is the full body of the test document about AI and graphs.")
    return doc


def _make_processor(backend) -> MetadataProcessor:
    router = MagicMock()
    router.select.return_value = backend
    registry = MagicMock()
    registry.get.side_effect = KeyError("no fallback registered")
    return MetadataProcessor(
        router=router,
        registry=registry,
        fallback_backend_name="none",
        max_retries=0,
    )


# ── Step 1: Body Load ──────────────────────────────────────────────────────────


def test_step1_body_load_from_file(monkeypatch, tmp_path):
    """save_body() + load_body() roundtrip from the flat-file body store."""
    monkeypatch.setattr("dewie.storage.body_store.BODIES_DIR", tmp_path / "bodies")
    doc_id = str(uuid.uuid4())
    text = "Full article body text stored on disk."
    save_body(doc_id, text)
    result = load_body(doc_id)
    assert result == text


def test_step1_body_load_redis_fallback(monkeypatch, tmp_path):
    """When body file is absent, the Redis value is used as fallback (pipeline pattern)."""
    monkeypatch.setattr("dewie.storage.body_store.BODIES_DIR", tmp_path / "bodies")
    body_from_file = load_body(str(uuid.uuid4()))  # None — file does not exist
    redis_value = "content retrieved from redis"
    content = body_from_file or redis_value or ""
    assert content == "content retrieved from redis"


def test_step1_body_load_empty(monkeypatch, tmp_path):
    """When no file and no Redis value, content degrades gracefully to empty string."""
    monkeypatch.setattr("dewie.storage.body_store.BODIES_DIR", tmp_path / "bodies")
    body_from_file = load_body(str(uuid.uuid4()))
    redis_value = None
    content = body_from_file or redis_value or ""
    assert content == ""


# ── Step 2: LLM Extraction ────────────────────────────────────────────────────


async def test_step2_llm_extraction_success():
    """Cache miss with pg → backend called, result cached; doc fully enriched."""
    backend = PassthroughBackend(name="test", response_json=VALID_EXTRACTION_JSON)
    processor = _make_processor(backend)
    doc = _make_doc()

    pg = AsyncMock()
    with (
        patch("dewie.enrichment.processor.get_cached", new=AsyncMock(return_value=None)),
        patch("dewie.enrichment.processor.set_cached", new=AsyncMock()),
    ):
        result_doc = await processor.enrich(doc, pg=pg)

    assert result_doc.status == ContentStatus.READY
    assert result_doc.summary == "A guide to AI and graph traversal systems."
    assert result_doc.document_type == DocumentType.BLOG_POST
    assert result_doc.tone == "informative"
    assert result_doc.author == "Alice Smith"
    assert "ai" in result_doc.keywords
    assert result_doc.sentiment == pytest.approx(0.4)
    assert result_doc.reading_level == ReadingLevel.STANDARD
    assert result_doc.language == "en"
    assert result_doc.answers_questions == ["What is graph traversal?"]


async def test_step2_llm_extraction_cache_hit():
    """Cache hit → backend NOT called; cached response used to populate doc."""
    backend_mock = MagicMock()
    backend_mock.name = "test"
    backend_mock.complete = AsyncMock()

    processor = _make_processor(backend_mock)
    doc = _make_doc()

    pg = AsyncMock()
    with (
        patch(
            "dewie.enrichment.processor.get_cached",
            new=AsyncMock(return_value=VALID_EXTRACTION_JSON),
        ),
        patch("dewie.enrichment.processor.set_cached", new=AsyncMock()) as mock_set,
    ):
        result_doc = await processor.enrich(doc, pg=pg)

    # Backend was NOT called — response came from cache
    backend_mock.complete.assert_not_called()
    # set_cached also NOT called — no new response to store
    mock_set.assert_not_called()
    # Doc is still fully enriched from the cached JSON
    assert result_doc.status == ContentStatus.READY
    assert result_doc.document_type == DocumentType.BLOG_POST
    assert result_doc.summary == "A guide to AI and graph traversal systems."


def test_step2_llm_extraction_malformed_json():
    """_parse_extraction_result raises ValueError on unparseable output."""
    garbage = "this is definitely not json !@#$%^ no curly braces at all"
    with pytest.raises((ValueError, json.JSONDecodeError)):
        _parse_extraction_result(garbage)


# ── Step 3: Field Population ──────────────────────────────────────────────────


def test_step3_field_population():
    """_apply_result_to_doc maps all ExtractionResult fields onto ContentDocument."""
    result = ExtractionResult.model_validate_json(VALID_EXTRACTION_JSON)
    doc = _make_doc()
    _apply_result_to_doc(doc, result)

    assert doc.document_type == DocumentType.BLOG_POST
    assert doc.author == "Alice Smith"
    assert doc.tone == "informative"
    assert doc.reading_level == ReadingLevel.STANDARD
    assert "ai" in doc.keywords
    assert "knowledge graphs" in doc.themes
    assert "OpenAI" in doc.entities
    assert doc.summary == "A guide to AI and graph traversal systems."
    assert doc.enrichment_quality_score == 75
    assert doc.sentiment == pytest.approx(0.4)
    assert doc.language == "en"


def test_step3_field_population_unknown_enum():
    """Unknown document_type from LLM is coerced to DocumentType.OTHER without crashing."""
    data = json.loads(VALID_EXTRACTION_JSON)
    data["document_type"] = "not_a_real_type_xyz_unknown"
    result = ExtractionResult.model_validate(data)
    doc = _make_doc()
    _apply_result_to_doc(doc, result)
    assert doc.document_type == DocumentType.OTHER


# ── Step 4: DB Upsert ─────────────────────────────────────────────────────────


async def test_step4_db_upsert_writes_all_fields():
    """enrich_and_persist() calls pg.upsert() with a doc containing all enrichment fields."""
    backend = PassthroughBackend(name="test", response_json=VALID_EXTRACTION_JSON)
    processor = _make_processor(backend)
    doc = _make_doc()

    pg = AsyncMock()
    pg.find_by_topics.return_value = []

    with (
        patch("dewie.enrichment.processor.get_cached", new=AsyncMock(return_value=None)),
        patch("dewie.enrichment.processor.set_cached", new=AsyncMock()),
        patch("dewie.pipeline.embed_batch", new=AsyncMock(return_value=None)),
    ):
        await processor.enrich_and_persist(doc, pg)

    pg.upsert.assert_called_once()
    upserted_doc: ContentDocument = pg.upsert.call_args[0][0]
    assert upserted_doc.document_type == DocumentType.BLOG_POST
    assert upserted_doc.sentiment == pytest.approx(0.4)
    assert upserted_doc.tone == "informative"
    assert upserted_doc.reading_level == ReadingLevel.STANDARD
    assert upserted_doc.author == "Alice Smith"
    # Topics are populated from themes when topics is empty
    assert len(upserted_doc.topics) > 0 or len(upserted_doc.themes) > 0


# ── Step 5: Embedding ─────────────────────────────────────────────────────────


def test_step5_embedding_text_build():
    """build_embed_text combines title, summary/embed_summary, and AQ into a dense string."""
    title = "Knowledge Graphs Explained"
    summary = "An overview of graph databases and their uses."
    aq = ["What is a knowledge graph?", "How are graphs traversed?"]
    body = "Knowledge graphs are structured networks of entities and relationships."

    # Without embed_summary: falls back to display summary; body is NOT included
    text = build_embed_text(title, summary, aq, body)
    assert title in text
    assert summary in text
    assert "What is a knowledge graph?" in text

    # With embed_summary: uses it instead of display summary
    embed_summary = (
        "Dense retrieval prose: knowledge graphs store entities and typed relationships."
    )
    text_dense = build_embed_text(title, summary, aq, body, embed_summary=embed_summary)
    assert title in text_dense
    assert embed_summary in text_dense
    assert summary not in text_dense  # embed_summary replaces display summary


async def test_step5_embedding_generation():
    """embed_batch result is stored via pg.set_embedding() with the correct doc_id and vector."""
    backend = PassthroughBackend(name="test", response_json=VALID_EXTRACTION_JSON)
    processor = _make_processor(backend)
    doc = _make_doc()

    fake_vector = [0.1, 0.2, 0.3, 0.4, 0.5]
    pg = AsyncMock()
    pg.find_by_topics.return_value = []

    with (
        patch("dewie.enrichment.processor.get_cached", new=AsyncMock(return_value=None)),
        patch("dewie.enrichment.processor.set_cached", new=AsyncMock()),
        patch("dewie.pipeline.embed_batch", new=AsyncMock(return_value=[fake_vector])),
    ):
        await processor.enrich_and_persist(doc, pg)

    pg.set_embedding.assert_called_once()
    called_doc_id, called_vector = pg.set_embedding.call_args[0]
    assert called_vector == fake_vector


async def test_step5_embedding_skipped_on_rate_limit():
    """embed_batch returning None (rate limit exhausted) does not mark the doc as FAILED."""
    from unittest.mock import MagicMock

    backend = PassthroughBackend(name="test", response_json=VALID_EXTRACTION_JSON)
    processor = _make_processor(backend)
    doc = _make_doc()

    mock_result = MagicMock()
    mock_result.scalar.return_value = 0
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=mock_result)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)
    mock_engine = MagicMock()
    mock_engine.begin.return_value = mock_conn

    pg = AsyncMock()
    pg._engine = mock_engine

    with (
        patch("dewie.enrichment.processor.get_cached", new=AsyncMock(return_value=None)),
        patch("dewie.enrichment.processor.set_cached", new=AsyncMock()),
        patch("dewie.pipeline.embed_batch", new=AsyncMock(return_value=None)),
    ):
        await processor.enrich_and_persist(doc, pg)

    # mark_status should only be called once — for PROCESSING — not FAILED
    pg.mark_status.assert_called_once_with(doc.id, ContentStatus.PROCESSING)
    pg.set_embedding.assert_not_called()


# ── Step 6: Relationship Building ─────────────────────────────────────────────


async def test_step6_relationships_built():
    """add_edges_for_doc is called for the enriched doc (SQL-native, no Python loop)."""
    from unittest.mock import MagicMock

    backend = PassthroughBackend(name="test", response_json=VALID_EXTRACTION_JSON)
    processor = _make_processor(backend)
    doc = _make_doc()

    mock_result = MagicMock()
    mock_result.scalar.return_value = 2
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=mock_result)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)
    mock_engine = MagicMock()
    mock_engine.begin.return_value = mock_conn

    pg = AsyncMock()
    pg._engine = mock_engine
    pg._is_sqlite = False

    with (
        patch("dewie.enrichment.processor.get_cached", new=AsyncMock(return_value=None)),
        patch("dewie.enrichment.processor.set_cached", new=AsyncMock()),
        patch("dewie.pipeline.embed_batch", new=AsyncMock(return_value=None)),
    ):
        await processor.enrich_and_persist(doc, pg)

    # add_edges_for_doc must have called conn.execute (SQL inverted-index path)
    assert mock_conn.execute.call_count >= 1


# ── Integration: Full pipeline debug trace ────────────────────────────────────


async def test_full_pipeline_single_doc_debug_trace(tmp_path, monkeypatch):
    """
    Run enrich_and_persist() with DEWIE_DEBUG=1.
    Assert all 6 debug step JSON files are written to the doc's debug directory.
    """
    import dewie.debug as dbg_module

    monkeypatch.setattr(dbg_module, "DEBUG", True)
    monkeypatch.setattr(dbg_module, "DEBUG_DIR", tmp_path / "dewie_debug")

    backend = PassthroughBackend(name="test", response_json=VALID_EXTRACTION_JSON)
    processor = _make_processor(backend)
    doc = _make_doc()

    from unittest.mock import MagicMock

    fake_vector = [0.1] * 10

    mock_result = MagicMock()
    mock_result.scalar.return_value = 0
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=mock_result)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)
    mock_engine = MagicMock()
    mock_engine.begin.return_value = mock_conn

    pg = AsyncMock()
    pg._engine = mock_engine
    pg._is_sqlite = False

    with (
        patch("dewie.enrichment.processor.get_cached", new=AsyncMock(return_value=None)),
        patch("dewie.enrichment.processor.set_cached", new=AsyncMock()),
        patch("dewie.pipeline.embed_batch", new=AsyncMock(return_value=[fake_vector])),
    ):
        await processor.enrich_and_persist(doc, pg)

    debug_dir = tmp_path / "dewie_debug" / str(doc.id)
    assert debug_dir.exists(), f"Debug dir not created: {debug_dir}"

    expected_steps = [
        "01_body_load.json",
        "02_llm_extraction.json",
        "03_field_population.json",
        "04_db_upsert.json",
        "05_embedding.json",
        "06_relationships.json",
    ]
    for step_file in expected_steps:
        path = debug_dir / step_file
        assert path.exists(), f"Missing debug step file: {step_file}"
        # Verify each file contains valid JSON
        data = json.loads(path.read_text())
        assert isinstance(data, dict), f"{step_file} does not contain a JSON object"


# ── save_raw_documents feature ────────────────────────────────────────────────

from dewie.config import Settings


def test_save_raw_documents_default_false():
    """save_raw_documents defaults to False."""
    assert Settings().save_raw_documents is False


async def test_save_raw_documents_writes_file(monkeypatch, tmp_path):
    """When save_raw_documents is True, the raw body is written to disk."""
    from unittest.mock import MagicMock

    from dewie.enrichment import processor as processor_module

    # Change working directory so relative paths resolve under tmp_path
    monkeypatch.chdir(tmp_path)

    # Mock settings to return True
    mock_settings = Settings()
    mock_settings.save_raw_documents = True
    monkeypatch.setattr(processor_module, "settings", mock_settings)

    backend = PassthroughBackend(name="test", response_json=VALID_EXTRACTION_JSON)
    processor = _make_processor(backend)
    doc = _make_doc(source="test-source")

    mock_result = MagicMock()
    mock_result.scalar.return_value = 0
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=mock_result)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)
    mock_engine = MagicMock()
    mock_engine.begin.return_value = mock_conn

    pg = AsyncMock()
    pg._engine = mock_engine

    with (
        patch("dewie.enrichment.processor.get_cached", new=AsyncMock(return_value=None)),
        patch("dewie.enrichment.processor.set_cached", new=AsyncMock()),
        patch("dewie.pipeline.embed_batch", new=AsyncMock(return_value=None)),
    ):
        await processor.enrich_and_persist(doc, pg)

    # Verify file was written
    expected_path = tmp_path / "ingested_docs" / "test-source" / f"{doc.id}.txt"
    assert expected_path.exists(), f"Expected file not found: {expected_path}"
    content = expected_path.read_text()
    assert content == doc.body


async def test_save_raw_documents_false_no_write(monkeypatch, tmp_path):
    """When save_raw_documents is False (default), no file is written."""
    from unittest.mock import MagicMock

    from dewie.enrichment import processor as processor_module

    monkeypatch.chdir(tmp_path)

    mock_settings = Settings()
    mock_settings.save_raw_documents = False
    monkeypatch.setattr(processor_module, "settings", mock_settings)

    backend = PassthroughBackend(name="test", response_json=VALID_EXTRACTION_JSON)
    processor = _make_processor(backend)
    doc = _make_doc(source="test-source")

    mock_result = MagicMock()
    mock_result.scalar.return_value = 0
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=mock_result)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)
    mock_engine = MagicMock()
    mock_engine.begin.return_value = mock_conn

    pg = AsyncMock()
    pg._engine = mock_engine

    with (
        patch("dewie.enrichment.processor.get_cached", new=AsyncMock(return_value=None)),
        patch("dewie.enrichment.processor.set_cached", new=AsyncMock()),
        patch("dewie.pipeline.embed_batch", new=AsyncMock(return_value=None)),
    ):
        await processor.enrich_and_persist(doc, pg)

    # Verify no file was written
    expected_path = tmp_path / "ingested_docs" / "test-source" / f"{doc.id}.txt"
    assert not expected_path.exists(), f"File should not exist: {expected_path}"


async def test_save_raw_documents_creates_dirs(monkeypatch, tmp_path):
    """Output directory is created automatically if it does not exist."""
    from unittest.mock import MagicMock

    from dewie.enrichment import processor as processor_module

    monkeypatch.chdir(tmp_path)

    mock_settings = Settings()
    mock_settings.save_raw_documents = True
    monkeypatch.setattr(processor_module, "settings", mock_settings)

    backend = PassthroughBackend(name="test", response_json=VALID_EXTRACTION_JSON)
    processor = _make_processor(backend)
    doc = _make_doc(source="nested/source")

    mock_result = MagicMock()
    mock_result.scalar.return_value = 0
    mock_conn = AsyncMock()
    mock_conn.execute = AsyncMock(return_value=mock_result)
    mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_conn.__aexit__ = AsyncMock(return_value=False)
    mock_engine = MagicMock()
    mock_engine.begin.return_value = mock_conn

    pg = AsyncMock()
    pg._engine = mock_engine

    assert not (tmp_path / "ingested_docs").exists()

    with (
        patch("dewie.enrichment.processor.get_cached", new=AsyncMock(return_value=None)),
        patch("dewie.enrichment.processor.set_cached", new=AsyncMock()),
        patch("dewie.pipeline.embed_batch", new=AsyncMock(return_value=None)),
    ):
        await processor.enrich_and_persist(doc, pg)

    expected_path = tmp_path / "ingested_docs" / "nested" / "source" / f"{doc.id}.txt"
    assert expected_path.exists(), f"Expected file not found: {expected_path}"
