"""Smoke test for the embedding stack against real OpenRouter + Postgres.

Proves the full vector path that unit tests cannot: real embeddings
from OpenRouter, real Vector-column writes to Postgres, the
register_vector round-trip (stored vector reads back as a float list),
and a cosine-distance similarity query returning sane ordering.

This is the first code that writes a real vector through pgvector, so
it is the verification for embed-on-approve and the seed of the search
tool: the query here is the pattern that tool will wrap.

Seeds four LearnedItem-shaped embedding records on distinct topics,
embeds and stores them, reads one back to confirm the round-trip, then
runs three query texts and prints the ranked hits for a human to
eyeball. A query about lists should rank the lists item first, etc.

Leaves its rows in the DB so they can be inspected via psql or the
search endpoint. Run `uv run python -m cli.admin db reset -y` then
`uv run python scripts/seed_domains.py` to clear.

Requires:
  - OPENROUTER_API_KEY in .env or the process environment.
  - Postgres up (docker compose up -d) and migrated (alembic upgrade head).

Run from backend/ with:
    uv run python scripts/smoke_embedding.py
"""

from __future__ import annotations

import asyncio

from app.core.config import get_settings
from app.core.db import SessionLocal
from app.models import Embedding, EmbeddingSourceType
from app.models.embedding import EMBEDDING_DIM
from app.services.embedding_service import (
    EmbeddingError,
    EmbeddingRecord,
    OpenRouterEmbedder,
    embed_records,
)
from sqlalchemy import select

# Four distinct topics. Each query below should rank its matching
# item first if retrieval works.
SEED_TEXTS: list[tuple[str, str]] = [
    (
        "What method appends one element to the end of a Python list?",
        "list.append(x) adds x to the end in place and returns None.",
    ),
    (
        "How do you define an async function in Python?",
        "Use 'async def name():'. Calling it returns a coroutine you await.",
    ),
    (
        "What does a database index do?",
        "An index speeds lookups on indexed columns at the cost of write overhead and storage.",
    ),
    (
        "What is the purpose of a Dockerfile?",
        "A Dockerfile is a recipe of instructions to build a container image.",
    ),
]

# (query text, which seed index it should rank first) for eyeballing.
QUERIES: list[tuple[str, int]] = [
    ("how to add an item to a list in python", 0),
    ("defining asynchronous functions", 1),
    ("why use an index on a table", 2),
]

TOP_K = 3


async def run() -> None:
    settings = get_settings()

    async with OpenRouterEmbedder(
        api_key=settings.openrouter_api_key.get_secret_value(),
        model=settings.openrouter_embedding_model,
    ) as embedder:
        print(f"Embedder model: {embedder.model_version}\n")

        records = [
            EmbeddingRecord(
                source_type=EmbeddingSourceType.LEARNED_ITEM,
                source_id=f"smoke-item-{i}",
                content=f"{q}\n{a}",
            )
            for i, (q, a) in enumerate(SEED_TEXTS)
        ]

        with SessionLocal() as db:
            print(f"Embedding and storing {len(records)} items...")
            rows = await embed_records(db=db, embedder=embedder, records=records)
            db.commit()
            print(f"  stored {len(rows)} embedding rows.\n")

            # Round-trip check: read one back, confirm the vector came
            # back as a list of the right width. This is what proves
            # register_vector is wired correctly, without it the column
            # reads back as a string.
            stored = db.execute(
                select(Embedding).where(Embedding.source_id == "smoke-item-0")
            ).scalar_one()
            vec = stored.embedding
            print("Round-trip check on smoke-item-0:")
            print(f"  type: {type(vec).__name__}, length: {len(vec)}")
            if len(vec) != EMBEDDING_DIM:
                raise RuntimeError(
                    f"Round-trip failed: vector width {len(vec)}, expected {EMBEDDING_DIM}."
                )
            if not all(isinstance(x, float) for x in vec[:8]):
                raise RuntimeError("Round-trip failed: vector elements are not floats.")
            print("  [check] vector round-trips as a float list.\n")

            # Similarity queries. cosine_distance ascending => nearest
            # first. 1 - distance is a 0..1 similarity score for display.
            print("Similarity queries:")
            all_ok = True
            for query_text, expected_idx in QUERIES:
                query_vec = (await embedder.embed_texts([query_text]))[0]
                hits = db.execute(
                    select(
                        Embedding.source_id,
                        Embedding.content,
                        Embedding.embedding.cosine_distance(query_vec).label("distance"),
                    )
                    .order_by("distance")
                    .limit(TOP_K)
                ).all()

                top_source_id = hits[0].source_id
                expected_source_id = f"smoke-item-{expected_idx}"
                ok = top_source_id == expected_source_id
                all_ok = all_ok and ok
                marker = "matches" if ok else "MISMATCH"
                print(f'  query: "{query_text}"')
                print(f"     expected top: {expected_source_id}  got: {top_source_id}  [{marker}]")
                for hit in hits:
                    score = 1 - hit.distance
                    snippet = hit.content.split("\n", 1)[0][:60]
                    print(f"       {hit.source_id}  score={score:.3f}  {snippet}")
                print()

            if all_ok:
                print("All queries ranked the expected item first.")
            else:
                print("Some queries did not rank as expected; eyeball the scores above.")

    print("\nSmoke complete. Rows left in DB for inspection.")
    print(
        "Clear with: uv run python -m cli.admin db reset -y && "
        "uv run python scripts/seed_domains.py"
    )


def main() -> None:
    try:
        asyncio.run(run())
    except EmbeddingError as e:
        print(f"\nEmbedding error: {e.message}")
        if e.cause is not None:
            print(f"Caused by: {type(e.cause).__name__}: {e.cause}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
