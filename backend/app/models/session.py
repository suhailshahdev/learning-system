"""Learning sessions.

A Session is one continuous learning block. It starts when the
user picks a topic (or lets Claude pick one) and ends when Claude
proposes to stop and the user agrees. Sessions can cover multiple
topics, so topic_id is optional.

Retests create a new session linked to the original rather than
modifying it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, ForeignKey, Integer, String
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, Timestamps, UUIDPrimaryKey
from app.models.enums import LearningMode, SessionState

if TYPE_CHECKING:
    from app.models.learned_item import LearnedItem
    from app.models.session_turn import SessionTurn
    from app.models.topic import Topic


class Session(Base, UUIDPrimaryKey, Timestamps):
    """One learning session."""

    __tablename__ = "session"

    topic_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("topic.id", ondelete="SET NULL"), nullable=True
    )
    mode_used: Mapped[LearningMode] = mapped_column(
        SQLEnum(LearningMode, native_enum=False, length=32),
        nullable=False,
    )
    state: Mapped[SessionState] = mapped_column(
        SQLEnum(SessionState, native_enum=False, length=32),
        nullable=False,
        default=SessionState.IN_PROGRESS,
    )
    claude_chat_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    claude_chat_message_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    parent_session_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("session.id", ondelete="SET NULL"), nullable=True
    )
    active_preferences: Mapped[list[Any]] = mapped_column(JSON, nullable=False, default=list)
    context_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    topic: Mapped[Topic | None] = relationship("Topic")
    parent_session: Mapped[Session | None] = relationship(
        "Session", remote_side="Session.id", back_populates="retests"
    )
    retests: Mapped[list[Session]] = relationship("Session", back_populates="parent_session")
    turns: Mapped[list[SessionTurn]] = relationship(
        "SessionTurn",
        back_populates="session",
        order_by="SessionTurn.turn_index",
        cascade="all, delete-orphan",
    )
    learned_items: Mapped[list[LearnedItem]] = relationship(
        "LearnedItem",
        back_populates="session",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Session id={self.id!r} state={self.state.value!r}>"
