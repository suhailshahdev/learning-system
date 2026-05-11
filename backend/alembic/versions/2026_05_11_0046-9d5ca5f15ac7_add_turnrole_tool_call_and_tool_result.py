"""add TurnRole tool_call and tool_result

Revision ID: 9d5ca5f15ac7
Revises: ee483ace1607
Create Date: 2026-05-11 00:46:55.946068+00:00
"""

from collections.abc import Sequence
from typing import Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "9d5ca5f15ac7"
down_revision: Union[str, None] = "ee483ace1607"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Non-native enum emits as VARCHAR(32) on SQLite with no CHECK
    # constraint actually enforced. This migration re-declares the
    # enum so future autogenerate sees no drift between the model
    # and migration history. No real SQL change happens on SQLite.
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
                "TRANSITION",
                "TOOL_CALL",
                "TOOL_RESULT",
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
                "TOOL_CALL",
                "TOOL_RESULT",
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
