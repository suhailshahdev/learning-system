"""Seed the domain table with the starting list.

Idempotent: existing rows by name are left untouched, missing rows
are inserted. Safe to re-run after extending the list.

Run from backend/ with:
    uv run python scripts/seed_domains.py
"""

from __future__ import annotations

from app.core.db import SessionLocal
from app.models import Domain, DomainKind

SEED: list[tuple[str, DomainKind]] = [
    ("Python", DomainKind.LANGUAGE),
    ("TypeScript", DomainKind.LANGUAGE),
    ("JavaScript", DomainKind.LANGUAGE),
    ("React", DomainKind.FRAMEWORK),
    ("FastAPI", DomainKind.FRAMEWORK),
    ("SQLAlchemy", DomainKind.LIBRARY),
    ("Node.js", DomainKind.TOOL),
    ("Git", DomainKind.TOOL),
    ("Docker", DomainKind.TOOL),
    ("System Design", DomainKind.CONCEPT),
    ("Databases", DomainKind.CONCEPT),
    ("HTTP & APIs", DomainKind.CONCEPT),
    ("Testing", DomainKind.PRACTICE),
]


def seed() -> None:
    """Insert any missing seed domains. Existing rows are left alone."""
    inserted = 0
    skipped = 0
    with SessionLocal() as session:
        existing_names = {name for (name,) in session.query(Domain.name).all()}
        for name, kind in SEED:
            if name in existing_names:
                print(f"exists:   {name}")
                skipped += 1
                continue
            session.add(Domain(name=name, kind=kind))
            print(f"inserted: {name} ({kind.value})")
            inserted += 1
        session.commit()
    print(f"\ndone. inserted={inserted} skipped={skipped} total={len(SEED)}")


if __name__ == "__main__":
    seed()
