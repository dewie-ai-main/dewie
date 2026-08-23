"""Tests for the save_raw_documents config option and file persistence."""

from __future__ import annotations

from uuid import uuid4

from dewie.config import Settings
from dewie.enrichment.processor import _save_raw_document
from dewie.models.content import ContentDocument


def test_save_raw_documents_default_false():
    """The save_raw_documents setting defaults to False."""
    assert Settings().save_raw_documents is False


def test_save_raw_documents_writes_file(tmp_path, monkeypatch):
    """When save_raw_documents=True, a file is written at the expected path."""
    from dewie.config import settings as config_settings

    monkeypatch.setattr(config_settings, "save_raw_documents", True)
    monkeypatch.chdir(tmp_path)

    doc = ContentDocument(
        id=uuid4(),
        url="https://example.com/test",
        title="Test",
        source="example.com",
        body="Hello raw document content",
    )
    _save_raw_document(doc)

    expected_file = tmp_path / "ingested_docs" / "example.com" / f"{doc.id}.txt"
    assert expected_file.exists()
    assert expected_file.read_text() == "Hello raw document content"


def test_save_raw_documents_false_no_write(tmp_path, monkeypatch):
    """When save_raw_documents=False, no files are written."""
    from dewie.config import settings as config_settings

    monkeypatch.setattr(config_settings, "save_raw_documents", False)
    monkeypatch.chdir(tmp_path)

    doc = ContentDocument(
        id=uuid4(),
        url="https://example.com/test",
        title="Test",
        source="example.com",
        body="Hello raw document content",
    )
    _save_raw_document(doc)

    assert not (tmp_path / "ingested_docs").exists()


def test_save_raw_documents_creates_dirs(tmp_path, monkeypatch):
    """Output directory is auto-created if it does not exist."""
    from dewie.config import settings as config_settings

    monkeypatch.setattr(config_settings, "save_raw_documents", True)
    monkeypatch.chdir(tmp_path)

    deep_source = "a/b/c/nested"
    doc = ContentDocument(
        id=uuid4(),
        url="https://example.com/deep",
        title="Deep",
        source=deep_source,
        body="nested body",
    )
    _save_raw_document(doc)

    expected_file = tmp_path / "ingested_docs" / deep_source / f"{doc.id}.txt"
    assert expected_file.exists()
