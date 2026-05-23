"""Semantic search over the embedding corpus.

Embeds a query string and returns the nearest stored embeddings by
cosine distance. Shared by the /api/search endpoint and, later, the
in-session retrieval tool. The cosine query is the one proven by the
embedding smoke: order by embedding.cosine_distance(query_vector),
ascending, limit k.

Read-only. Does not commit. The embedder is passed in so the caller
controls its lifecycle (app-scoped singleton in the API).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from sqlalchemy import select

from app.models import Embedding

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession

    from app.models import EmbeddingSourceType
    from app.services.embedding_service import Embedder


# Default and ceiling for how many hits a search returns. The ceiling
# stops a caller (or the LLM, later) from pulling the whole corpus.
DEFAULT_LIMIT = 10
MAX_LIMIT = 50


@dataclass(frozen=True)
class SearchHit:
    """One search result: the matched embedding's source and score.

    score is cosine similarity in 0..1 (1 - cosine_distance), higher
    is more similar. source_type and source_id identify what was
    matched so the caller can resolve back to the learned item or
    document. content is the embedded text, returned directly so the
    caller need not re-fetch the source.
    """

    source_type: EmbeddingSourceType
    source_id: str
    content: str
    score: float


async def search_corpus(
    db: DbSession,
    embedder: Embedder,
    query: str,
    *,
    limit: int = DEFAULT_LIMIT,
    source_type: EmbeddingSourceType | None = None,
) -> list[SearchHit]:
    """Embed the query and return the nearest stored embeddings.

    limit is clamped to MAX_LIMIT. source_type filters to one corpus
    kind (learned items vs document chunks); None searches all. An
    empty query or empty corpus returns an empty list. Raises
    EmbeddingError if the query cannot be embedded.
    """
    if not query.strip():
        return []

    effective_limit = min(limit, MAX_LIMIT)

    query_vector = (await embedder.embed_texts([query]))[0]

    stmt = select(
        Embedding.source_type,
        Embedding.source_id,
        Embedding.content,
        Embedding.embedding.cosine_distance(query_vector).label("distance"),
    )
    if source_type is not None:
        stmt = stmt.where(Embedding.source_type == source_type)
    stmt = stmt.order_by("distance").limit(effective_limit)

    rows = db.execute(stmt).all()
    return [
        SearchHit(
            source_type=row.source_type,
            source_id=row.source_id,
            content=row.content,
            score=1.0 - row.distance,
        )
        for row in rows
    ]
