"""Topics view route.

One GET endpoint returning the composed domains and topics tree.
The service does the work and the route is a thin pass-through.
"""

from __future__ import annotations

from fastapi import APIRouter

# FastAPI resolves Annotated dependencies at route registration by
# evaluating annotation strings against the module's runtime
# namespace. The dependency alias must be a real import and cannot
# be TYPE_CHECKING-only.
from app.api.deps import DbSession  # noqa: TC001
from app.schemas.topics import TopicsResponse
from app.services.topics_service import build_topics_response

router = APIRouter(prefix="/topics", tags=["topics"])


@router.get("", response_model=TopicsResponse)
async def get_topics(db: DbSession) -> TopicsResponse:
    """Return all domains plus the flat topic list."""
    return await build_topics_response(db)
