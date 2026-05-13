"""add learned_item grading_verdict

Revision ID: 968b44af7db0
Revises: b4ec5a1e7a9f
Create Date: 2026-05-12 11:21:27.853610+00:00
"""

from collections.abc import Sequence
from typing import Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "968b44af7db0"
down_revision: Union[str, None] = "b4ec5a1e7a9f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Real schema change: adds the column nullable with no default.
    # Existing rows get NULL, which is accurate since they predate the
    # split-roundtrip feature and have no grading verdict to record.
    with op.batch_alter_table("learned_item", schema=None) as batch_op:
        batch_op.add_column(
            sa.Column(
                "grading_verdict",
                sa.Enum(
                    "CORRECT",
                    "PARTIAL",
                    "INCORRECT",
                    "OPEN_GRADED",
                    name="gradingverdict",
                    native_enum=False,
                    length=32,
                ),
                nullable=True,
            )
        )


def downgrade() -> None:
    with op.batch_alter_table("learned_item", schema=None) as batch_op:
        batch_op.drop_column("grading_verdict")
