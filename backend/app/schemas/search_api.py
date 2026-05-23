"""Request and response schemas for the search endpoint.

Takes a search query, finds the closest matching items in the
database, and returns them ranked by how closely they match.
The request carries the query and an optional result limit.
The response carries the matched items with their match scores.
"""

from __future__ import annotations

from app.models.enums import EmbeddingSourceType  # noqa: TC002
from pydantic import BaseModel, ConfigDict, Field


class SearchRequest(BaseModel):
    """Query for POST /api/search."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query: str = Field(min_length=1, max_length=2000, description="The search query text.")
    limit: int = Field(default=10, ge=1, le=50, description="Max hits to return.")


class SearchHitResponse(BaseModel):
    """One ranked hit in the search response."""

    model_config = ConfigDict(frozen=True)

    source_type: EmbeddingSourceType
    source_id: str
    content: str
    score: float


class SearchResponse(BaseModel):
    """Ranked hits for a search query, most similar first."""

    model_config = ConfigDict(frozen=True)

    query: str
    hits: list[SearchHitResponse]
