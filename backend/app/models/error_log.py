"""Errors surfaced by the system.

An `ErrorLog` row captures one error: where it came from, what
went wrong, and any context useful for debugging. Services write
to this table whenever an operation fails in a way the user
needs to know about.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import JSON, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, Timestamps, UUIDPrimaryKey


class ErrorLog(Base, UUIDPrimaryKey, Timestamps):
    """One logged error."""

    __tablename__ = "error_log"

    session_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("session.id", ondelete="SET NULL"), nullable=True, index=True
    )
    kind: Mapped[str] = mapped_column(String(128), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    def __repr__(self) -> str:
        return f"<ErrorLog kind={self.kind!r} session_id={self.session_id!r}>"
