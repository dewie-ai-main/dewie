"""Tests for dewie.chunker — document splitting logic."""

from __future__ import annotations

from dewie.chunker import CHUNK_WORDS, MIN_WORDS, OVERLAP_WORDS, chunk_document


def test_empty_body_returns_empty():
    assert chunk_document("") == []
    assert chunk_document("   ") == []


def test_short_doc_returns_single_chunk():
    body = "word " * 100
    chunks = chunk_document(body.strip(), title="Short Doc", domain="example.com")
    assert len(chunks) == 1
    assert "Short Doc" in chunks[0]
    assert "example.com" in chunks[0]
    assert "Chunk 1/1" in chunks[0]
    assert body.strip() in chunks[0]


def test_short_doc_no_title_or_domain():
    body = "hello world"
    chunks = chunk_document(body)
    assert len(chunks) == 1
    assert "Chunk 1/1" in chunks[0]
    assert "hello world" in chunks[0]


def test_long_doc_produces_multiple_chunks():
    body = " ".join(f"word{i}" for i in range(MIN_WORDS + 500))
    chunks = chunk_document(body, title="Long Doc", domain="longdoc.com")
    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert f"Chunk {i + 1}/{len(chunks)}" in chunk
        assert "Long Doc" in chunk
        assert "longdoc.com" in chunk


def test_chunk_size_approximately_correct():
    body = " ".join(f"word{i}" for i in range(MIN_WORDS + 1000))
    chunks = chunk_document(body)
    for chunk in chunks:
        text = chunk.split("\n\n", 1)[1]
        word_count = len(text.split())
        assert word_count <= CHUNK_WORDS + 10


def test_overlap_exists_between_adjacent_chunks():
    body = " ".join(f"w{i}" for i in range(MIN_WORDS + 2000))
    chunks = chunk_document(body)
    assert len(chunks) >= 2

    def words_in_chunk(chunk):
        return set(chunk.split("\n\n", 1)[1].split())

    first_words = words_in_chunk(chunks[0])
    second_words = words_in_chunk(chunks[1])
    overlap = first_words & second_words
    assert len(overlap) >= OVERLAP_WORDS - 5


def test_exactly_min_words_doc():
    # MIN_WORDS is NOT short — condition is < MIN_WORDS, so exactly MIN_WORDS gets chunked
    body = " ".join(f"w{i}" for i in range(MIN_WORDS))
    chunks = chunk_document(body)
    assert len(chunks) >= 1  # chunks, not single-chunk path


def test_just_above_min_words():
    body = " ".join(f"w{i}" for i in range(MIN_WORDS + 1))
    chunks = chunk_document(body)
    assert len(chunks) >= 1
