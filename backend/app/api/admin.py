"""Admin / observability HTTP routes.

Read-only endpoints over the llm_call table: a filtered list of
recorded round-trips and an aggregate stats summary. These power
the observability page.

"admin" is a path naming convention, not an access boundary. The
system has no auth, these routes are reachable by anyone who can
reach the API, which is localhost only. If auth is ever added,
the admin-prefixed routes are the natural place to gate.

Pure DB reads: no transport dependencies, no service-error
translation beyond what the read functions raise (they raise
nothing today, an empty table is an empty list, not an error).
"""

from __future__ import annotations

from fastapi import APIRouter, Query

# FastAPI resolves Annotated[...] dependency aliases at route
# registration against the module's runtime namespace, so the dep
# alias import must be runtime, not TYPE_CHECKING-only.
from app.api.deps import DbSession  # noqa: TC001
from app.models.enums import TransportKind  # noqa: TC001
from app.schemas.admin_api import LLMCallListResponse, LLMCallStats
from app.services.admin_service import (
    DEFAULT_STATS_WINDOW_DAYS,
    list_llm_calls,
    llm_call_stats,
)

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/llm-calls", response_model=LLMCallListResponse)
async def list_calls(
    db: DbSession,
    transport_kind: TransportKind | None = None,
    success: bool | None = None,
) -> LLMCallListResponse:
    """List recorded LLM round-trips, newest first.

    Optional filters: transport_kind narrows to one transport,
    success=true/false narrows to successes or failures. Capped at
    the service's row limit, limit_reached signals more exist.
    """
    return list_llm_calls(db=db, transport_kind=transport_kind, success=success)


@router.get("/llm-calls/stats", response_model=LLMCallStats)
async def call_stats(
    db: DbSession,
    window_days: int = Query(default=DEFAULT_STATS_WINDOW_DAYS, ge=1, le=365),
    transport_kind: TransportKind | None = None,
) -> LLMCallStats:
    """Aggregate stats over LLM calls in the last window_days.

    Call count, error rate, latency p50/p95, and total cost over
    the window, optionally filtered by transport. window_days is
    clamped to 1..365.
    """
    return llm_call_stats(db=db, window_days=window_days, transport_kind=transport_kind)
