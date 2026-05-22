"""add pgvector extension, document and embedding tables

Revision ID: 79329541d3ad
Revises: 968b44af7db0
Create Date: 2026-05-22 05:00:34.066937+00:00
"""

from collections.abc import Sequence
from typing import Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = "79329541d3ad"
down_revision: Union[str, None] = "968b44af7db0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # The extension must exist before any vector column is created.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "document",
        sa.Column("title", sa.String(length=256), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "embedding",
        sa.Column(
            "source_type",
            sa.Enum(
                "LEARNED_ITEM",
                "DOCUMENT_CHUNK",
                name="embeddingsourcetype",
                native_enum=False,
                length=32,
            ),
            nullable=False,
        ),
        sa.Column("source_id", sa.String(length=36), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=False),
        sa.Column("embedding_model_version", sa.String(length=64), nullable=False),
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )

    # HNSW chosen over ivfflat because embeddings are written
    # incrementally (one per approved item, chunks on document
    # ingest), never bulk-loaded. ivfflat trains its centroids on
    # existing rows, so building it here against an empty table
    # produces a broken index. HNSW needs no training and adapts as
    # rows arrive. Revisit only at large scale (millions of rows).
    # Cosine distance pairs with the normalized OpenAI embeddings.
    op.create_index(
        "ix_embedding_hnsw",
        "embedding",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_index("ix_embedding_source", "embedding", ["source_type", "source_id"])


def downgrade() -> None:
    op.drop_index("ix_embedding_source", table_name="embedding")
    op.drop_index("ix_embedding_hnsw", table_name="embedding")
    op.drop_table("embedding")
    op.drop_table("document")
    op.execute("DROP EXTENSION IF EXISTS vector")
