"""Request and response schemas for the document ingest endpoint."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class IngestDocumentRequest(BaseModel):
    """Body for POST /api/documents."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str = Field(min_length=1, max_length=256, description="Short label for the document.")
    content: str = Field(min_length=1, description="The full document text to ingest.")


class IngestDocumentResponse(BaseModel):
    """Result of ingesting a document."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    title: str
    chunk_count: int
