"""rename TurnRole.CLAUDE to ASSISTANT

Revision ID: b8d38ee093bb
Revises: db8dabc9fef6
Create Date: 2026-04-27 14:30:56.690185+00:00
"""

from collections.abc import Sequence
from typing import Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b8d38ee093bb"
down_revision: Union[str, None] = "db8dabc9fef6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # The non-native enum type emits as VARCHAR(32) on SQLite with no
    # CHECK constraint actually enforced (see handover doc section 8).
    # This migration re-declares the enum so future autogenerate runs
    # see no drift between the model and migration history. No real
    # SQL change happens on SQLite.
    with op.batch_alter_table("session_turn", schema=None) as batch_op:
        batch_op.alter_column(
            "role",
            existing_type=sa.Enum(
                "CLAUDE",
                "USER",
                "SYSTEM",
                name="turnrole",
                native_enum=False,
                length=32,
            ),
            type_=sa.Enum(
                "ASSISTANT",
                "USER",
                "SYSTEM",
                name="turnrole",
                native_enum=False,
                length=32,
            ),
            existing_nullable=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("session_turn", schema=None) as batch_op:
        batch_op.alter_column(
            "role",
            existing_type=sa.Enum(
                "ASSISTANT",
                "USER",
                "SYSTEM",
                name="turnrole",
                native_enum=False,
                length=32,
            ),
            type_=sa.Enum(
                "CLAUDE",
                "USER",
                "SYSTEM",
                name="turnrole",
                native_enum=False,
                length=32,
            ),
            existing_nullable=False,
        )
