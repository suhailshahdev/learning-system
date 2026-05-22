"""Vector embeddings for semantic retrieval.

One row per embedded piece of text. The embedding is a 1536-dim
vector from the configured embedding provider. source_type and
source_id together point back at what produced the text: a
learned_item or a document. content holds the exact text that was
embedded, denormalized so a retrieval hit returns its text without
joining back to the source, and so the vector and its text stay
together even if the source row is later edited.

embedding_model_version records which model produced the vector.
When the provider or model changes, a mismatch here marks rows that
need re-embedding.
"""

from __future__ import annotations

from pgvector.sqlalchemy import Vector
from sqlalchemy import Enum as SQLEnum
from sqlalchemy import Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, Timestamps, UUIDPrimaryKey
from app.models.enums import EmbeddingSourceType

# text-embedding-3-small output dimension. The embedding column is
# fixed at this width. Changing the provider to a different dimension
# is a migration, not a config flip.
EMBEDDING_DIM = 1536


class Embedding(Base, UUIDPrimaryKey, Timestamps):
    """One embedded text with its source reference and vector."""

    __tablename__ = "embedding"

    # Indexes declared here so model metadata matches the migration.
    # Otherwise autogenerate sees them only in the live DB and tries
    # to drop them. The hnsw index needs pgvector's cosine ops, which
    # a column-level index=True cannot express.
    __table_args__ = (
        Index(
            "ix_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_embedding_source", "source_type", "source_id"),
    )

    source_type: Mapped[EmbeddingSourceType] = mapped_column(
        SQLEnum(EmbeddingSourceType, native_enum=False, length=32),
        nullable=False,
    )
    # Polymorphic reference: a learned_item.id or a document.id
    # depending on source_type. Not a ForeignKey because it targets
    # two tables. Resolved in the service layer, not by the database.
    source_id: Mapped[str] = mapped_column(String(36), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    embedding_model_version: Mapped[str] = mapped_column(String(64), nullable=False)

    def __repr__(self) -> str:
        return f"<Embedding source_type={self.source_type.value!r} source_id={self.source_id!r}>"
