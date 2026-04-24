"""Turns within a session.

A `SessionTurn` is one message exchange: a question from Claude,
an answer from the user, or a system message.
Turns are ordered within a session by `turn_index`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import JSON, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, Timestamps, UUIDPrimaryKey
from app.models.enums import LearningMode, TurnRole

if TYPE_CHECKING:
    from app.models.session import Session


class SessionTurn(Base, UUIDPrimaryKey, Timestamps):
    """One turn within a session."""

    __tablename__ = "session_turn"
    __table_args__ = (UniqueConstraint("session_id", "turn_index", name="uq_session_turn_order"),)

    session_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("session.id", ondelete="CASCADE"), nullable=False
    )
    turn_index: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[TurnRole] = mapped_column(
        SQLEnum(TurnRole, native_enum=False, length=32),
        nullable=False,
    )
    raw_content: Mapped[str] = mapped_column(Text, nullable=False)
    parsed: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    mode: Mapped[LearningMode | None] = mapped_column(
        SQLEnum(LearningMode, native_enum=False, length=32),
        nullable=True,
    )

    session: Mapped[Session] = relationship("Session", back_populates="turns")

    def __repr__(self) -> str:
        return (
            f"<SessionTurn session_id={self.session_id!r} "
            f"index={self.turn_index} role={self.role.value!r}>"
        )
