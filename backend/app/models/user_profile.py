"""Singleton user profile.

One row per install. Holds the user's name, option resume and
target job description text, default stack choice, Chrome profile
path used by the Playwright transport, and a record of which
prerequisite warnings the user has already dismissed

Singleton is enforced at the schema level by a fixed primary key.
Any attempt to insert a second row fails on the unique PK, so the
"exactly one row" invariant is a data rule, not an application
discipline.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, Timestamps

SINGLETON_ID = "00000000-0000-0000-0000-000000000001"
"""Fixed primary key for the single user_profile row. Code that loads
theprofile should use this constant rather than querying by other
fields."""


class UserProfile(Base, Timestamps):
    """The single user_profile row for this install."""

    __tablename__ = "user_profile"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=SINGLETON_ID,
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    default_stack: Mapped[str | None] = mapped_column(String(64), nullable=True)
    resume_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    target_jd_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    chrome_profile_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    prereq_warning_dismissed: Mapped[list[Any]] = mapped_column(JSON, nullable=False, default=list)

    def __repr__(self) -> str:
        return f"<UserProfile name={self.name!r}>"
