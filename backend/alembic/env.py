"""Alembic runtime entry point.

This script is invoked by the `alembic` CLI for every migration operation.
It connects Alembic to the application's SQLAlchemy engine and metadata.

Why we override the default:
- The database URL comes from app.core.config (which reads .env), not
  from alembic.ini. Single source of truth for configuration.
- target_metadata points at app.models.Base.metadata so autogenerate
  can diff against our declared models. Importing app.models pulls in
  every model module, registering each table on the metadata before
  autogenerate scans it.
"""

from logging.config import fileConfig

from alembic import context
from app.core.config import get_settings
from app.models import Base
from sqlalchemy import engine_from_config, pool

config = context.config

# Inject the runtime DB URL. alembic.ini leaves this blank on purpose.
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database_url)

# Configure logging from alembic.ini.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Emits SQL to stdout instead of connecting to the database. Useful
    for generating migration scripts to run manually on a server we
    cannot reach directly.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    Connects to the database and applies migrations directly. This is
    the common path for local development.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
            render_as_batch=connection.dialect.name == "sqlite",
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
