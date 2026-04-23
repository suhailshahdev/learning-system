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
    """Adds a UUID string primary key generated in Python.

    Stored as `String(36)`, which is the UUID written out with
    hyphens (for example, `550e8400-e29b-41d4-a716-446655440000`).
    This shape works on both SQLite and Postgres. When we move to
    Postgres later, a migration can switch the column to the
    native `UUID` type without changing any application code.
    """

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )


class Timestamps:
    """Adds `created_at` and `updated_at` as timezone-aware UTC.

    `created_at` is set once when the row is first saved.
    `updated_at` is set again every time the row is changed,
    through SQLAlchemy's `onupdate` hook.
    """

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
        nullable=False,
    )
