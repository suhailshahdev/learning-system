"""User-defined teaching preferences.

A `TeachingPreference` is one short instruction the user wants
Claude to follow. Examples: "use real-world analogies", "keep
code examples minimal", "explain the why before the what".

At the start of every session, the system sends Claude the set
of preferences marked active. The `is_active` flag lets the
user turn a preference off without deleting it.
"""

from __future__ import annotations

from sqlalchemy import Boolean, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, Timestamps, UUIDPrimaryKey


class TeachingPreference(Base, UUIDPrimaryKey, Timestamps):
    __tablename__ = "teaching_preference"
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    def __repr__(self) -> str:
        return f"<TeachingPreference name={self.name!r} active={self.is_active}>"
