"""Document ingest endpoint.

One POST endpoint: ingest a pasted text document into the retrieval
corpus. Thin pass-through to document_service.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.api.deps import DbSession, EmbedderDep  # noqa: TC001
from app.schemas.document_api import IngestDocumentRequest, IngestDocumentResponse
from app.services.document_service import ingest_document
from app.services.embedding_service import EmbeddingError

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=IngestDocumentResponse, status_code=status.HTTP_201_CREATED)
async def ingest(
    body: IngestDocumentRequest,
    db: DbSession,
    embedder: EmbedderDep,
) -> IngestDocumentResponse:
    """Chunk, embed, and store a pasted text document."""
    try:
        document, chunk_count = await ingest_document(
            db=db,
            embedder=embedder,
            title=body.title,
            content=body.content,
        )
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except EmbeddingError as exc:
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY,
            detail=f"Embedding the document failed: {exc.message}",
        ) from exc

    return IngestDocumentResponse(
        document_id=document.id,
        title=document.title,
        chunk_count=chunk_count,
    )
