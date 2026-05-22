"""Database engine and session factory.

The engine is built once per process. Sessions are short-lived and
opened for a single piece of work. In the API layer, each request
gets its own session through FastAPI's dependency injection.

This file only sets up the basics. Models and the declarative base
live in app.models. Migrations live in alembic/.
"""

from collections.abc import Generator

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings


def _build_engine() -> Engine:
    """Construct the SQLAlchemy engine from current settings.

    Separated from module-level initialisation so tests can reconstruct
    the engine against a different database by clearing the settings
    cache and calling _build_engine() again
    """
    settings = get_settings()

    connect_args: dict[str, object] = {}
    if settings.database_url.startswith("sqlite"):
        # SQLite forbids sharing a connection across threads by default.
        # FastAPI may run things on different threads, so we turn the
        # check off. Safe here because only one thing writes at a time.
        connect_args["check_same_thread"] = False

    new_engine = create_engine(
        settings.database_url, connect_args=connect_args, echo=False, future=True
    )

    # pgvector needs each psycopg connection told about the vector type
    # so Vector columns round-trip as Python lists. Only Postgres has
    # the type. The SQLite fallback and the in-memory test engine must
    # not run this.
    if settings.database_url.startswith("postgresql"):
        from pgvector.psycopg import register_vector  # noqa: PLC0415

        @event.listens_for(new_engine, "connect")
        def _register_vector_on_connect(dbapi_connection: object, _record: object) -> None:
            register_vector(dbapi_connection)

    return new_engine


engine: Engine = _build_engine()

SessionLocal: sessionmaker[Session] = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


def get_db() -> Generator[Session]:
    """Yield a database session for the duration of a request.

    Used as a FastAPI dependency. The session is closed automatically
    when the request handler returns, whether or not an exception was
    raised.
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
