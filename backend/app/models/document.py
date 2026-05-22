"""Ingested text documents for the retrieval corpus.

A `Document` is a chunk of free text the user pasted in: notes, an
article, reference material. It exists to give retrieval a corpus
beyond the user's own learned items, so semantic search returns
useful hits and not just paraphrases of past questions.

The original full text lives here. The embeddable slices of it live
as `embedding` rows with source_type=document_chunk pointing back at
this row's id. Chunks have no table of their own: a chunk is just
its text plus its parent document, both of which the embedding row
already carries.
"""

from __future__ import annotations

from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, Timestamps, UUIDPrimaryKey


class Document(Base, UUIDPrimaryKey, Timestamps):
    """One ingested text document. Source for document-chunk embeddings."""

    __tablename__ = "document"

    title: Mapped[str] = mapped_column(String(256), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    def __repr__(self) -> str:
        return f"<Document title={self.title!r}>"
