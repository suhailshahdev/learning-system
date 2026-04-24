"""Topic tree.

A `Topic` is one node in a tree of subjects. Paths take the form
`Domain > Category > Subtopic`. The tree grows as Claude
introduces new topics during sessions. Domains live in the
`domain` table; root topics within a domain have `parent_id = NULL`.
"""

from __future__ import annotations

from datetime import (
    datetime,  # noqa: TC003  (SQLAlchemy re-evaluates Mapped[datetime | None] at runtime)
)
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, String
from sqlalchemy import Enum as SQLEnum
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, Timestamps, UUIDPrimaryKey
from app.models.enums import Difficulty, TopicStatus


class Topic(Base, UUIDPrimaryKey, Timestamps):
    """One node in the topic tree."""

    __tablename__ = "topic"

    path: Mapped[str] = mapped_column(String(512), unique=True, nullable=False)
    domain: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    parent_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("topic.id", ondelete="SET NULL"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    difficulty: Mapped[Difficulty | None] = mapped_column(
        SQLEnum(Difficulty, native_enum=False, length=32),
        nullable=True,
    )
    prerequisites: Mapped[list[Any]] = mapped_column(JSON, nullable=False, default=list)
    status: Mapped[TopicStatus] = mapped_column(
        SQLEnum(TopicStatus, native_enum=False, length=32),
        nullable=False,
        default=TopicStatus.NOT_STARTED,
    )
    last_reviewed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    tags: Mapped[list[Any]] = mapped_column(JSON, nullable=False, default=list)

    parent: Mapped[Topic | None] = relationship(
        "Topic", remote_side="Topic.id", back_populates="children"
    )
    children: Mapped[list[Topic]] = relationship(
        "Topic", back_populates="parent", cascade="save-update, merge"
    )

    def __repr__(self) -> str:
        return f"<Topic path={self.path!r} status={self.status.value!r}>"
