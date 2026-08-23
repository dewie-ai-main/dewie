"""Tests for dewie.storage.body_store — file I/O helpers."""

from __future__ import annotations

import tempfile
from unittest.mock import patch


def _with_temp_dir():
    return tempfile.mkdtemp()


# ── save_body / load_body ─────────────────────────────────────────────────────


def test_save_and_load_body(tmp_path):
    from dewie.storage import body_store

    doc_id = "abcd1234-0000-0000-0000-000000000000"
    with patch.object(body_store, "_bodies_dir", return_value=tmp_path / "bodies"):
        body_store.save_body(doc_id, "Hello world")
        result = body_store.load_body(doc_id)
    assert result == "Hello world"


def test_save_body_skips_empty(tmp_path):
    from dewie.storage import body_store

    doc_id = "abcd1234-0000-0000-0000-000000000001"
    with patch.object(body_store, "_bodies_dir", return_value=tmp_path / "bodies"):
        body_store.save_body(doc_id, "")
        result = body_store.load_body(doc_id)
    assert result is None


def test_save_body_skips_whitespace_only(tmp_path):
    from dewie.storage import body_store

    doc_id = "abcd1234-0000-0000-0000-000000000002"
    with patch.object(body_store, "_bodies_dir", return_value=tmp_path / "bodies"):
        body_store.save_body(doc_id, "   \n  ")
        result = body_store.load_body(doc_id)
    assert result is None


def test_load_body_missing_returns_none(tmp_path):
    from dewie.storage import body_store

    with patch.object(body_store, "_bodies_dir", return_value=tmp_path / "bodies"):
        result = body_store.load_body("00000000-0000-0000-0000-000000000099")
    assert result is None


def test_body_exists_true(tmp_path):
    from dewie.storage import body_store

    doc_id = "abcd1234-0000-0000-0000-000000000003"
    with patch.object(body_store, "_bodies_dir", return_value=tmp_path / "bodies"):
        body_store.save_body(doc_id, "content")
        assert body_store.body_exists(doc_id) is True


def test_body_exists_false(tmp_path):
    from dewie.storage import body_store

    with patch.object(body_store, "_bodies_dir", return_value=tmp_path / "bodies"):
        assert body_store.body_exists("00000000-0000-0000-0000-999999999999") is False


def test_delete_body(tmp_path):
    from dewie.storage import body_store

    doc_id = "abcd1234-0000-0000-0000-000000000004"
    with patch.object(body_store, "_bodies_dir", return_value=tmp_path / "bodies"):
        body_store.save_body(doc_id, "to delete")
        assert body_store.body_exists(doc_id) is True
        body_store.delete_body(doc_id)
        assert body_store.body_exists(doc_id) is False


def test_delete_body_missing_ok(tmp_path):
    from dewie.storage import body_store

    with patch.object(body_store, "_bodies_dir", return_value=tmp_path / "bodies"):
        # Should not raise
        body_store.delete_body("00000000-0000-0000-0000-000000000000")


def test_bodies_dir_function():
    from dewie.storage.body_store import bodies_dir

    result = bodies_dir()
    assert result.name == "bodies"


def test_save_body_uuid_type(tmp_path):
    from uuid import UUID

    from dewie.storage import body_store

    doc_id = UUID("abcd1234-5678-0000-0000-000000000005")
    with patch.object(body_store, "_bodies_dir", return_value=tmp_path / "bodies"):
        body_store.save_body(doc_id, "uuid content")
        result = body_store.load_body(doc_id)
    assert result == "uuid content"
