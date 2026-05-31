"""Schemas for the LLM-call observability HTTP API.

GET /api/admin/llm-calls returns a filtered, capped list of
recorded LLM round-trips. GET /api/admin/llm-calls/stats returns
aggregates over a time window: call count, error rate, latency
percentiles, and total cost.

The list is a flat projection of the llm_call table, the same
explicit-projection approach the sessions browse uses: internal
columns added to LLMCall do not auto-leak into the response.

"admin" in the path is a naming convention, not an access
boundary. The system has no auth layer, these routes are reachable
by anyone who can reach the API, which is localhost only.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003

# Pydantic v2 resolves field type annotations at validation time,
# so the enum import stays at runtime, not TYPE_CHECKING-only.
from app.models.enums import TransportKind  # noqa: TC002
from pydantic import BaseModel, ConfigDict


class LLMCallRow(BaseModel):
    """One recorded LLM round-trip in the browse list.

    Token and cost fields are nullable: claude.ai reports no
    tokens, DeepSeek's usage is not yet threaded, and cost is
    unknown until pricing lands. They surface as null rather than
    zero so the UI can distinguish "no data" from "zero".
    """

    model_config = ConfigDict(frozen=True)

    id: str
    trace_id: str
    session_id: str | None
    transport_kind: TransportKind
    method: str
    model: str | None
    latency_ms: int
    prompt_chars: int
    response_chars: int
    prompt_tokens: int | None
    completion_tokens: int | None
    cost_usd: float | None
    success: bool
    error: str | None
    created_at: datetime


class LLMCallListResponse(BaseModel):
    """Response for GET /api/admin/llm-calls.

    rows sorted by created_at desc. limit_reached signals more
    rows exist past the cap, same convention as the sessions
    browse.
    """

    model_config = ConfigDict(frozen=True)

    rows: list[LLMCallRow]
    limit_reached: bool


class LLMCallStats(BaseModel):
    """Aggregates over LLM calls in a time window.

    Computed over the last `window_days` of calls, optionally
    filtered by transport. Percentiles are over latency_ms across
    all calls in the window. error_rate is failures over total.
    total_cost_usd sums the non-null cost_usd values. It is 0.0
    when no call in the window has a known cost, which is the
    current state until pricing lands.
    """

    model_config = ConfigDict(frozen=True)

    window_days: int
    total_calls: int
    error_count: int
    error_rate: float
    latency_p50_ms: int | None
    latency_p95_ms: int | None
    total_cost_usd: float
