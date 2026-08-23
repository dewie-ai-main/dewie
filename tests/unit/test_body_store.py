"""Tests for dewie.storage.body_store."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4


def _patch_bodies_dir(monkeypatch, tmp_path: Path) -> Path:
    bodies = tmp_path / "bodies"
    monkeypatch.setattr("dewie.storage.body_store._bodies_dir", lambda: bodies)
    return bodies


# ── save_body ─────────────────────────────────────────────────────────────────


def test_save_body_writes_file(monkeypatch, tmp_path):
    from dewie.storage.body_store import save_body

    _patch_bodies_dir(monkeypatch, tmp_path)
    doc_id = str(uuid4())
    save_body(doc_id, "Hello, world!")
    path = tmp_path / "bodies" / doc_id[:2] / f"{doc_id}.txt"
    assert path.exists()
    assert path.read_text() == "Hello, world!"


def test_save_body_skips_empty(monkeypatch, tmp_path):
    from dewie.storage.body_store import save_body

    bodies = _patch_bodies_dir(monkeypatch, tmp_path)
    doc_id = str(uuid4())
    save_body(doc_id, "")
    path = bodies / doc_id[:2] / f"{doc_id}.txt"
    assert not path.exists()


def test_save_body_skips_whitespace_only(monkeypatch, tmp_path):
    from dewie.storage.body_store import save_body

    bodies = _patch_bodies_dir(monkeypatch, tmp_path)
    doc_id = str(uuid4())
    save_body(doc_id, "   \n  ")
    path = bodies / doc_id[:2] / f"{doc_id}.txt"
    assert not path.exists()


def test_save_body_accepts_uuid_object(monkeypatch, tmp_path):
    from dewie.storage.body_store import save_body

    _patch_bodies_dir(monkeypatch, tmp_path)
    doc_id = uuid4()
    save_body(doc_id, "content")
    path = tmp_path / "bodies" / str(doc_id)[:2] / f"{doc_id}.txt"
    assert path.exists()


def test_save_body_shards_by_prefix(monkeypatch, tmp_path):
    from dewie.storage.body_store import save_body

    bodies = _patch_bodies_dir(monkeypatch, tmp_path)
    doc_id = "abcdef-1234"
    save_body(doc_id, "body text")
    shard_dir = bodies / "ab"
    assert shard_dir.is_dir()
    assert (shard_dir / f"{doc_id}.txt").exists()


def test_save_body_handles_write_error(monkeypatch, tmp_path, caplog):
    from dewie.storage.body_store import save_body

    bodies = _patch_bodies_dir(monkeypatch, tmp_path)
    # Make directory read-only to trigger write failure
    bodies.mkdir(parents=True, exist_ok=True)
    shard = bodies / "ab"
    shard.mkdir()
    shard.chmod(0o444)
    try:
        import logging

        with caplog.at_level(logging.WARNING, logger="dewie.storage.body_store"):
            save_body("abcdef", "content")
        # Should not raise
    finally:
        shard.chmod(0o755)


# ── load_body ─────────────────────────────────────────────────────────────────


def test_load_body_returns_content(monkeypatch, tmp_path):
    from dewie.storage.body_store import load_body, save_body

    _patch_bodies_dir(monkeypatch, tmp_path)
    doc_id = str(uuid4())
    save_body(doc_id, "test content")
    result = load_body(doc_id)
    assert result == "test content"


def test_load_body_returns_none_if_missing(monkeypatch, tmp_path):
    from dewie.storage.body_store import load_body

    _patch_bodies_dir(monkeypatch, tmp_path)
    result = load_body(str(uuid4()))
    assert result is None


def test_load_body_accepts_uuid_object(monkeypatch, tmp_path):
    from dewie.storage.body_store import load_body, save_body

    _patch_bodies_dir(monkeypatch, tmp_path)
    doc_id = uuid4()
    save_body(doc_id, "uuid content")
    result = load_body(doc_id)
    assert result == "uuid content"


# ── body_exists ───────────────────────────────────────────────────────────────


def test_body_exists_true(monkeypatch, tmp_path):
    from dewie.storage.body_store import body_exists, save_body

    _patch_bodies_dir(monkeypatch, tmp_path)
    doc_id = str(uuid4())
    save_body(doc_id, "content")
    assert body_exists(doc_id) is True


def test_body_exists_false(monkeypatch, tmp_path):
    from dewie.storage.body_store import body_exists

    _patch_bodies_dir(monkeypatch, tmp_path)
    assert body_exists(str(uuid4())) is False


# ── delete_body ───────────────────────────────────────────────────────────────


def test_delete_body_removes_file(monkeypatch, tmp_path):
    from dewie.storage.body_store import body_exists, delete_body, save_body

    _patch_bodies_dir(monkeypatch, tmp_path)
    doc_id = str(uuid4())
    save_body(doc_id, "to be deleted")
    assert body_exists(doc_id) is True
    delete_body(doc_id)
    assert body_exists(doc_id) is False


def test_delete_body_missing_does_not_raise(monkeypatch, tmp_path):
    from dewie.storage.body_store import delete_body

    _patch_bodies_dir(monkeypatch, tmp_path)
    # Should not raise
    delete_body(str(uuid4()))


# ── bodies_dir ────────────────────────────────────────────────────────────────


def test_bodies_dir_returns_path(monkeypatch, tmp_path):
    from dewie.storage.body_store import bodies_dir

    monkeypatch.setattr("dewie.storage.body_store._bodies_dir", lambda: tmp_path / "bodies")
    result = bodies_dir()
    assert result == tmp_path / "bodies"
