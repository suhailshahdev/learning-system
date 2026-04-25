"""Typer CLI for admin operations.

Run from backend/ with:
    uv run python -m cli.admin <command>

Two commands are implemented: `db inspect` lists row counts per
table, and `db reset` wipes the database back to a fresh schema.
"""

from __future__ import annotations

from pathlib import Path

import typer
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from app.core.db import SessionLocal
from app.models import Base
from sqlalchemy import func, select, text

app = typer.Typer(help="Admin CLI for the learning system.")
db_app = typer.Typer(help="Database operations.")
session_app = typer.Typer(help="Session inspection.")
topic_app = typer.Typer(help="Topic tree inspection.")
error_app = typer.Typer(help="Error log inspection.")

app.add_typer(db_app, name="db")
app.add_typer(session_app, name="session")
app.add_typer(topic_app, name="topic")
app.add_typer(error_app, name="error")


def _alembic_config() -> AlembicConfig:
    """Build an Alembic config pointing at this project's alembic.ini."""
    ini_path = Path(__file__).resolve().parent.parent / "alembic.ini"
    return AlembicConfig(str(ini_path))


@db_app.command("inspect")
def db_inspect() -> None:
    """Print row counts for every table."""
    counts: dict[str, int] = {}
    with SessionLocal() as session:
        for name, table in Base.metadata.tables.items():
            stmt = select(func.count()).select_from(table)
            counts[name] = session.execute(stmt).scalar_one()
        counts["alembic_version"] = session.execute(
            text("SELECT COUNT(*) FROM alembic_version")
        ).scalar_one()

    width = max(len(name) for name in counts)
    typer.echo(f"{'Table'.ljust(width)}   Count")
    typer.echo(f"{'-' * width}   -----")
    for name in sorted(counts):
        typer.echo(f"{name.ljust(width)}   {counts[name]:>5}")


@db_app.command("reset")
def db_reset(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Wipe the database and re-apply migrations from scratch."""
    if not yes:
        confirmation = typer.prompt(
            "This will wipe all data. Type RESET to confirm",
            default="",
            show_default=False,
        )
        if confirmation != "RESET":
            typer.echo("Aborted.")
            raise typer.Exit(code=1)

    config = _alembic_config()
    typer.echo("Downgrading to base...")
    alembic_command.downgrade(config, "base")
    typer.echo("Upgrading to head...")
    alembic_command.upgrade(config, "head")
    typer.echo("Database reset.")
    typer.echo("Run `uv run python scripts/seed_domains.py` to repopulate domains.")


@db_app.command("export")
def db_export(path: Path) -> None:
    """Dump the database to a JSON file."""
    raise NotImplementedError


@db_app.command("import")
def db_import(path: Path) -> None:
    """Restore the database from a JSON file."""
    raise NotImplementedError


@session_app.command("list")
def session_list() -> None:
    """List sessions with their state."""
    raise NotImplementedError


@session_app.command("show")
def session_show(session_id: str) -> None:
    """Print the full transcript of a session."""
    raise NotImplementedError


@topic_app.command("tree")
def topic_tree() -> None:
    """Print the topic tree."""
    raise NotImplementedError


@error_app.command("log")
def error_log() -> None:
    """Tail recent errors from the error log."""
    raise NotImplementedError


if __name__ == "__main__":
    app()
