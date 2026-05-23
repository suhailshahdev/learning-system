"""Backfill embeddings for existing learned items.

Embeds every LearnedItem that does not yet have an embedding row and
writes the rows in batches, reporting progress. Idempotent: items
already embedded are skipped, so the script is safe to re-run after
adding more items or after an interrupted run.

Skips by presence of any embedding row for the item, not by model
version. Re-embedding when the embedding model changes is a separate
concern. The embedding_model_version column supports it but this
script does not act on it. When that's needed, add a --remodel flag
that targets rows whose version differs from the current model.

Embeds in batches so a large corpus commits incrementally rather than
holding everything until the end, and so a failure partway leaves the
already-committed batches in place (re-run resumes from there).

Run from backend/ with:
    uv run python scripts/backfill_embeddings.py

Requires:
  - OPENROUTER_API_KEY in .env or the process environment.
  - Postgres up and migrated.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from app.core.config import get_settings
from app.core.db import SessionLocal
from app.models import Embedding, EmbeddingSourceType, LearnedItem
from app.services.embedding_service import (
    EmbeddingError,
    OpenRouterEmbedder,
    embed_records,
    records_from_learned_items,
)
from sqlalchemy import select

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession

# Items embedded per batch. Each batch is one OpenRouter request (well
# under MAX_INPUTS_PER_REQUEST) and one commit. Small enough that a
# failure loses little, large enough to amortize the round trip.
BATCH_SIZE = 64


def _unembedded_items(db: DbSession) -> list[LearnedItem]:
    """Return learned items that have no embedding row yet.

    A learned item is embedded when an embedding row exists with
    source_type=LEARNED_ITEM and source_id equal to the item id.
    source_id is a polymorphic string reference, not a foreign key,
    so this is a NOT IN against that column rather than a join.
    """
    embedded_ids = select(Embedding.source_id).where(
        Embedding.source_type == EmbeddingSourceType.LEARNED_ITEM
    )
    stmt = (
        select(LearnedItem)
        .where(LearnedItem.id.not_in(embedded_ids))
        .order_by(LearnedItem.created_at)
    )
    return list(db.execute(stmt).scalars().all())


async def run() -> None:
    settings = get_settings()

    with SessionLocal() as db:
        items = _unembedded_items(db)
        total = len(items)
        if total == 0:
            print("Nothing to backfill: all learned items already embedded.")
            return
        print(f"Backfilling {total} learned items in batches of {BATCH_SIZE}.\n")

        async with OpenRouterEmbedder(
            api_key=settings.openrouter_api_key.get_secret_value(),
            model=settings.openrouter_embedding_model,
        ) as embedder:
            done = 0
            for start in range(0, total, BATCH_SIZE):
                batch = items[start : start + BATCH_SIZE]
                records = records_from_learned_items(batch)
                try:
                    await embed_records(db=db, embedder=embedder, records=records)
                    db.commit()
                except EmbeddingError as e:
                    db.rollback()
                    print(f"\nFailed on batch starting at item {start}: {e.message}")
                    print(f"Committed {done} items before the failure. Re-run to resume.")
                    raise SystemExit(1) from e

                done += len(batch)
                print(f"  embedded {done}/{total}")

        print(f"\nDone. Embedded {done} items.")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
