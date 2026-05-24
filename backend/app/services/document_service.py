"""Document ingest for the retrieval corpus.

Takes a pasted text document, splits it into chunks, embeds each
chunk, and stores the document plus one embedding row per chunk.
This is what makes retrieval more than search over the user's own
questions: pasted notes and articles become a corpus to ground on.

Embedding happens before the write, with no transaction open, so a
network call never holds a transaction. The document row and all its
chunk embeddings then commit together: a document with no chunks is
useless and an orphan chunk has no parent, so the write is atomic.
Unlike embed-on-approve, ingest is not best-effort. Embedding is the
point, so a failure fails the ingest and the caller surfaces it.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from app.models import Document, EmbeddingSourceType
from app.services.embedding_service import EmbeddingError, EmbeddingRecord, embed_records

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession

    from app.services.embedding_service import Embedder


# Chunk size cap in characters. Paragraphs under this become one chunk
# each, longer ones fall back to sentence splitting. ~1200 chars is
# roughly 300 tokens: coherent passages, precise enough as retrieval
# hits, well under the embedding model's per-input token limit. Tuning
# this trades retrieval precision against chunk count. Revisit if hits
# feel too broad or too granular.
MAX_CHUNK_CHARS = 1200

# Sentence boundary: a period, question mark, or exclamation mark
# followed by whitespace. Crude but adequate for prose. It is only
# used to break paragraphs that exceed the cap, not as the primary
# split. Edge cases (abbreviations, decimals) produce slightly odd
# breaks but never lose content.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str) -> list[str]:
    """Split text into embeddable chunks.

    Paragraph-first: split on blank lines, the author's natural
    boundaries. Any paragraph longer than MAX_CHUNK_CHARS falls back
    to sentence splitting, accumulating sentences into chunks under
    the cap. A single sentence longer than the cap is emitted whole
    rather than cut mid-word. Embedding tolerates it and cutting
    would lose meaning.

    Pure function: no DB, no network. Empty or whitespace-only input
    returns an empty list.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= MAX_CHUNK_CHARS:
            chunks.append(paragraph)
        else:
            chunks.extend(_split_oversized(paragraph))
    return chunks


def _split_oversized(paragraph: str) -> list[str]:
    """Break a paragraph over the cap into sentence-grouped chunks.

    Accumulates sentences until adding the next would exceed the cap,
    then starts a new chunk. A lone sentence over the cap is emitted
    as its own chunk uncut.
    """
    sentences = [s.strip() for s in _SENTENCE_BOUNDARY.split(paragraph) if s.strip()]

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= MAX_CHUNK_CHARS:
            current = f"{current} {sentence}"
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


async def ingest_document(
    db: DbSession,
    embedder: Embedder,
    *,
    title: str,
    content: str,
) -> tuple[Document, int]:
    """Ingest a document: chunk, embed, and store atomically.

    Chunks the content, embeds every chunk (network, no transaction
    open), then writes the Document row and one embedding row per
    chunk in a single transaction. Raises EmbeddingError if embedding
    fails, leaving nothing written. Raises ValueError if the content
    produces no chunks (empty or whitespace-only).
    """
    chunks = chunk_text(content)
    if not chunks:
        raise ValueError("Document content produced no chunks.")

    document = Document(title=title, content=content)
    db.add(document)
    db.flush()  # populate document.id for the chunk source_id

    records = [
        EmbeddingRecord(
            source_type=EmbeddingSourceType.DOCUMENT_CHUNK,
            source_id=document.id,
            content=chunk,
        )
        for chunk in chunks
    ]

    try:
        await embed_records(db=db, embedder=embedder, records=records)
    except EmbeddingError:
        # Embedding failed. The whole ingest fails atomically. Roll
        # back the document row added above so nothing is written.
        db.rollback()
        raise

    db.commit()
    db.refresh(document)
    return document, len(chunks)
