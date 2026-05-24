"""Tests for the document chunker.

Covers chunk_text's pure logic: paragraph splitting, the oversized-
paragraph sentence fallback, the lone-oversized-sentence case, and
empty input. ingest_document's storage path (embedding + atomic
write) is covered by the smoke against real Postgres, since it needs
the vector column.
"""

from __future__ import annotations

from app.services.document_service import MAX_CHUNK_CHARS, chunk_text


def test_chunk_text_splits_on_blank_lines() -> None:
    """Paragraphs separated by blank lines become separate chunks."""
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = chunk_text(text)
    assert chunks == ["First paragraph.", "Second paragraph.", "Third paragraph."]


def test_chunk_text_empty_returns_empty() -> None:
    """Empty or whitespace-only input produces no chunks."""
    assert chunk_text("") == []
    assert chunk_text("   \n\n  \n  ") == []


def test_chunk_text_collapses_multiple_blank_lines() -> None:
    """Runs of blank lines are one boundary, not several empty chunks."""
    text = "One.\n\n\n\nTwo."
    assert chunk_text(text) == ["One.", "Two."]


def test_chunk_text_oversized_paragraph_splits_on_sentences() -> None:
    """A paragraph over the cap is broken into sentence-grouped chunks."""
    sentence = "This is a sentence that takes up some space. "
    # Build a paragraph well over the cap with no blank lines.
    paragraph = (sentence * 60).strip()
    assert len(paragraph) > MAX_CHUNK_CHARS

    chunks = chunk_text(paragraph)

    assert len(chunks) > 1
    # Every chunk is under the cap (sentences are short enough to group).
    assert all(len(c) <= MAX_CHUNK_CHARS for c in chunks)
    # No content lost: every sentence's text survives across the chunks.
    rejoined = " ".join(chunks)
    assert rejoined.count("This is a sentence") == 60


def test_chunk_text_lone_oversized_sentence_emitted_whole() -> None:
    """A single sentence longer than the cap is kept whole, not cut."""
    huge = "word " * (MAX_CHUNK_CHARS // 2)  # one "sentence", no boundaries
    huge = huge.strip()
    assert len(huge) > MAX_CHUNK_CHARS

    chunks = chunk_text(huge)

    assert len(chunks) == 1
    assert chunks[0] == huge


def test_chunk_text_normal_paragraph_under_cap_is_one_chunk() -> None:
    """A normal paragraph under the cap stays a single chunk."""
    text = "A short coherent paragraph about Python lists and how append works."
    assert chunk_text(text) == [text]
