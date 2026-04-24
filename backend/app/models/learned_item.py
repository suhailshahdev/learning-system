"""Items the user has learned.

A `LearnedItem` is one question/answer pair that the user approved
at the end of a session. Before approval it lives in the session
turn. Only on approval does it become a learned item.

This is the source of truth for the review queue, the "what I
know" summary, and retest mode.
"""

from __future__ import annotations

from datetime import (
    datetime,  # noqa: TC003  (SQLAlchemy re-evaluates Mapped[datetime | None] at runtime)
)
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, Timestamps, UUIDPrimaryKey
from app.models.enums import Difficulty, LearnedItemStatus, LearningMode

if TYPE_CHECKING:
    from app.models.session import Session
    from app.models.topic import Topic


class LearnedItem(Base, UUIDPrimaryKey, Timestamps):
    """One approved question/answer pair pinned to a topic."""

    __tablename__ = "learned_item"

    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("session.id", ondelete="CASCADE"), nullable=False
    )
    topic_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("topic.id", ondelete="RESTRICT"), nullable=False
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    your_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    mode: Mapped[LearningMode] = mapped_column(
        SQLEnum(LearningMode, native_enum=False, length=32),
        nullable=False,
    )
    difficulty: Mapped[Difficulty | None] = mapped_column(
        SQLEnum(Difficulty, native_enum=False, length=32),
        nullable=True,
    )
    status: Mapped[LearnedItemStatus] = mapped_column(
        SQLEnum(LearnedItemStatus, native_enum=False, length=32),
        nullable=False,
        default=LearnedItemStatus.LEARNED,
    )
    last_reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    session: Mapped[Session] = relationship("Session", back_populates="learned_items")
    topic: Mapped[Topic] = relationship("Topic")

    def __repr__(self) -> str:
        return f"<LearnedItem topic_id={self.topic_id!r} status={self.status.value!r}>"
