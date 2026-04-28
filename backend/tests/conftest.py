"""Shared pytest fixtures.

The `db` fixture provides a fresh in-memory SQLite database per
test, with the full model schema created via Base.metadata. Tests
can write rows freely and the database is discarded at the end.

Bypasses Alembic intentionally: tests validate service and model
logic, not migrations. The Alembic-applied schema is exercised
via the real database in smoke scripts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from app.models import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

if TYPE_CHECKING:
    from collections.abc import Generator

    from sqlalchemy.orm import Session as DbSession


@pytest.fixture
def db() -> Generator[DbSession]:
    """Yield a SQLAlchemy session backed by a fresh in-memory SQLite DB.

    Each test gets its own database. The connection is shared across
    the session so that schema created by Base.metadata.create_all is
    visible to subsequent queries within the test.
    """
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)

    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    session = session_factory()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()
