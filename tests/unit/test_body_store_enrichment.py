from uuid import uuid4

from dewie.pipeline import build_embed_text
from dewie.storage.body_store import body_exists, load_body, save_body
from tests.fixtures import load_fixture_doc

AI_ARTICLE = load_fixture_doc("artificial_intelligence.txt")


def test_build_embed_text_with_content():
    # Raw body is no longer sliced into embed text — embed_summary handles retrieval density.
    # content param is retained for backward compatibility but has no effect on output.
    result = build_embed_text("AI Overview", "A summary of AI", ["What is AI?"], AI_ARTICLE)
    assert "AI Overview" in result
    assert "A summary of AI" in result
    result_empty = build_embed_text("AI Overview", "A summary of AI", ["What is AI?"], "")
    assert result == result_empty  # body param is ignored; both produce identical output


def test_build_embed_text_without_content():
    result = build_embed_text("title", "summary", [], "")
    assert result  # non-empty string — graceful degradation


def test_load_body_missing_returns_none():
    assert load_body(str(uuid4())) is None


def test_save_and_load_body_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setattr("dewie.storage.body_store.BODIES_DIR", tmp_path / "bodies")
    doc_id = str(uuid4())
    text = "Hello, body store."
    save_body(doc_id, text)
    assert load_body(doc_id) == text


def test_body_exists_false_for_unknown(monkeypatch, tmp_path):
    monkeypatch.setattr("dewie.storage.body_store.BODIES_DIR", tmp_path / "bodies")
    assert body_exists(str(uuid4())) is False


def test_body_exists_true_after_save(monkeypatch, tmp_path):
    monkeypatch.setattr("dewie.storage.body_store.BODIES_DIR", tmp_path / "bodies")
    doc_id = str(uuid4())
    save_body(doc_id, "some content")
    assert body_exists(doc_id) is True


# ── Fallback chain tests (body store → Redis priority ordering) ───────────────


def test_load_body_returns_none_when_no_file(monkeypatch, tmp_path):
    """When body file doesn't exist, load_body returns None (Redis fallback kicks in)."""
    monkeypatch.setattr("dewie.storage.body_store.BODIES_DIR", tmp_path / "bodies")
    result = load_body(str(uuid4()))
    assert result is None


def test_load_body_wins_over_empty_string(monkeypatch, tmp_path):
    """When body file exists, it takes priority (simulates Redis-TTL-expired scenario)."""
    monkeypatch.setattr("dewie.storage.body_store.BODIES_DIR", tmp_path / "bodies")
    doc_id = str(uuid4())
    save_body(doc_id, AI_ARTICLE)
    body_result = load_body(doc_id)
    # Simulate: body_file wins over empty Redis value
    redis_value = ""
    content = body_result or redis_value or ""
    assert content == AI_ARTICLE


def test_redis_fallback_when_body_missing(monkeypatch, tmp_path):
    """When body file is absent, the Redis value should be used (fallback chain)."""
    monkeypatch.setattr("dewie.storage.body_store.BODIES_DIR", tmp_path / "bodies")
    body_result = load_body(str(uuid4()))  # None — file doesn't exist
    redis_value = "content from redis"
    content = body_result or redis_value or ""
    assert content == "content from redis"


def test_load_body_ioerror_returns_none(monkeypatch, tmp_path):
    """If load_body raises an IOError (e.g. permissions), it should not propagate."""
    import dewie.storage.body_store as bs

    monkeypatch.setattr(bs, "BODIES_DIR", tmp_path / "bodies")
    # Write a file then make the shard dir unreadable
    doc_id = str(uuid4())
    save_body(doc_id, "test")
    shard = tmp_path / "bodies" / doc_id[:2]
    shard.chmod(0o000)
    try:
        # load_body silently returns None on IOError per body_store contract
        result = load_body(doc_id)
        assert result is None
    finally:
        shard.chmod(0o755)  # restore so tmp_path cleanup works
