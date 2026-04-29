"""add session transport_kind

Revision ID: ba986ad17982
Revises: b8d38ee093bb
Create Date: 2026-04-29 09:05:52.054071+00:00

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sqlalchemy as sa
from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence


# revision identifiers, used by Alembic.
revision: str = "ba986ad17982"
down_revision: str | None = "b8d38ee093bb"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add transport_kind to session.

    The column is non-nullable in the model. To support a clean
    migration over a populated table, the column is added with a
    transient server-side default so existing rows get backfilled,
    then the default is dropped. New INSERTs from the application
    must specify transport_kind explicitly.
    """
    with op.batch_alter_table("session") as batch:
        batch.add_column(
            sa.Column(
                "transport_kind",
                sa.String(length=32),
                nullable=False,
                server_default="claude_playwright",
            )
        )

    with op.batch_alter_table("session") as batch:
        batch.alter_column("transport_kind", server_default=None)


def downgrade() -> None:
    """Remove transport_kind from session."""
    with op.batch_alter_table("session") as batch:
        batch.drop_column("transport_kind")
