"""Search endpoint.

One POST endpoint: embed a query and return ranked corpus hits.
Thin pass-through to search_service, mapping an embedding failure to
502 (the upstream embedding provider failed).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.api.deps import DbSession, EmbedderDep  # noqa: TC001
from app.schemas.search_api import SearchHitResponse, SearchRequest, SearchResponse
from app.services.embedding_service import EmbeddingError
from app.services.search_service import search_corpus

router = APIRouter(prefix="/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search(
    body: SearchRequest,
    db: DbSession,
    embedder: EmbedderDep,
) -> SearchResponse:
    """Embed the query and return the nearest corpus hits."""
    try:
        hits = await search_corpus(
            db=db,
            embedder=embedder,
            query=body.query,
            limit=body.limit,
        )
    except EmbeddingError as exc:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            detail=f"Embedding the query failed: {exc.message}",
        ) from exc

    return SearchResponse(
        query=body.query,
        hits=[
            SearchHitResponse(
                source_type=hit.source_type,
                source_id=hit.source_id,
                content=hit.content,
                score=hit.score,
            )
            for hit in hits
        ],
    )
