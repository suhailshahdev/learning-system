"""add TurnRole.TRANSITION

Revision ID: ee483ace1607
Revises: ba986ad17982
Create Date: 2026-04-30 14:51:04.297636+00:00
"""

from collections.abc import Sequence
from typing import Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "ee483ace1607"
down_revision: Union[str, None] = "ba986ad17982"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # The non-native enum type emits as VARCHAR(32) on SQLite with no
    # CHECK constraint actually enforced. This migration re-declares
    # the enum so future autogenerate runs see no drift between the
    # model and migration history. No real SQL change happens on SQLite.

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
                "ASSISTANT",
                "USER",
                "SYSTEM",
                "TRANSITION",
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
                "TRANSITION",
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
