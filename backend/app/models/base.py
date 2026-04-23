"""Base class and shared mixins for all SQLAlchemy models.

`Base` is the declarative base every model registeres against. The two
mixins cover behaviour every domain table needs: a UUID primary key and
created/updated timestamps. Models compose them explicitly rather than
inheriting from a god-base, so the model definition line reads as a
description of what the model has.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base. Every model inherits from this."""


class UUIDPrimaryKey:
    """Adds a UUI string primary key generated Python-side

    Stored as `String(36)` (canonical UUID form with hypes) for
    cross-database compatibility; swappable to native `UUID` on
    Postgres via migration without changing application code.
    """

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )


class Timestamps:
    """Adds `created_at` and `updated_at` as timezone-aware UTC.

    `created_at` is set once at INSERT; `updated_at` is refreshed on
    every UPDATE via SQLAlchemy's `onupdate` hook.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )
