"""User knowledge assertions.

A `UserKnowledgeAssertion` records that the user knows a specific
topic at a specific difficulty, sourced from one of four places:
the user said so themselves, the system derived it from
learned items, it came from the resume, or it came from the
target job description.

These rows are the single source of truth for "does the user
know X?" The session engine consults them before teaching a
topic so it can warn about unmet prerequisites, and the
settings UI exposes them for manual edits.

The `topic_path` is stored as a string rather than a foreign key
to the `topic` table. Assertions can be created before the
matching topic row exists (for example, a resume mentions
Docker before any Docker session has happened).
"""

from __future__ import annotations

from sqlalchemy import Enum as SQLEnum
from sqlalchemy import String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, Timestamps, UUIDPrimaryKey
from app.models.enums import AssertionSource, Difficulty


class UserKnowledgeAssertion(Base, UUIDPrimaryKey, Timestamps):
    """One claim about what the user knows."""

    __tablename__ = "user_knowledge_assertion"

    topic_path: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    difficulty: Mapped[Difficulty] = mapped_column(
        SQLEnum(Difficulty, native_enum=False, length=32),
        nullable=False,
    )
    source: Mapped[AssertionSource] = mapped_column(
        SQLEnum(AssertionSource, native_enum=False, length=32),
        nullable=False,
    )
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<UserKnowledgeAssertion "
            f"topic_path={self.topic_path!r} "
            f"difficulty={self.difficulty.value!r} "
            f"source={self.source.value!r}>"
        )
