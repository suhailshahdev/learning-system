"""LLM-call observability read service.

Two read functions over the llm_call table: list_llm_calls returns
a capped, optionally-filtered list of rows, llm_call_stats computes
aggregates over a time window. Read-only, no transport calls, no
commits.

Percentiles are computed in Python rather than via a SQL percentile
function. percentile_cont is Postgres-only and the test database is
SQLite, so a SQL-side percentile would be untestable against the
unit suite. At single-user scale the row count in any window is
small, so pulling latencies and computing in Python is both correct
on either database and cheap. If this ever scales, the percentile
moves to percentile_cont on the Postgres path.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from sqlalchemy import select

from app.models import LLMCall, TransportKind
from app.schemas.admin_api import LLMCallListResponse, LLMCallRow, LLMCallStats

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession

# Maximum rows returned by the list endpoint. Same reasoning and
# limit+1 convention as the sessions browse: defer pagination until
# real friction, signal limit_reached when the cap is hit.
LLM_CALL_LIMIT = 100

# Default window for the stats aggregate when the caller does not
# specify one.
DEFAULT_STATS_WINDOW_DAYS = 7


def list_llm_calls(
    db: DbSession,
    *,
    transport_kind: TransportKind | None = None,
    success: bool | None = None,
) -> LLMCallListResponse:
    """Return llm_call rows, newest first, optionally filtered.

    transport_kind=None returns all transports, passing one filters
    to it. success=None returns all. True returns only successful
    calls, False only failures. Rows are capped at LLM_CALL_LIMIT
    with the limit+1 trick setting limit_reached.
    """
    stmt = select(LLMCall).order_by(LLMCall.created_at.desc())
    if transport_kind is not None:
        stmt = stmt.where(LLMCall.transport_kind == transport_kind)
    if success is not None:
        stmt = stmt.where(LLMCall.success == success)
    stmt = stmt.limit(LLM_CALL_LIMIT + 1)

    calls = list(db.execute(stmt).scalars().all())

    limit_reached = len(calls) > LLM_CALL_LIMIT
    visible = calls[:LLM_CALL_LIMIT]

    rows = [
        LLMCallRow(
            id=call.id,
            trace_id=call.trace_id,
            session_id=call.session_id,
            transport_kind=call.transport_kind,
            method=call.method,
            model=call.model,
            latency_ms=call.latency_ms,
            prompt_chars=call.prompt_chars,
            response_chars=call.response_chars,
            prompt_tokens=call.prompt_tokens,
            completion_tokens=call.completion_tokens,
            cost_usd=call.cost_usd,
            success=call.success,
            error=call.error,
            created_at=call.created_at,
        )
        for call in visible
    ]
    return LLMCallListResponse(rows=rows, limit_reached=limit_reached)


def llm_call_stats(
    db: DbSession,
    *,
    window_days: int = DEFAULT_STATS_WINDOW_DAYS,
    transport_kind: TransportKind | None = None,
) -> LLMCallStats:
    """Compute aggregates over calls in the last window_days.

    Counts, error rate, latency p50/p95, and summed cost over all
    calls whose created_at is within the window, optionally filtered
    by transport. Percentiles are over latency_ms, they are None when
    the window has no calls.
    """
    cutoff = datetime.now(UTC) - timedelta(days=window_days)

    stmt = select(LLMCall.latency_ms, LLMCall.success, LLMCall.cost_usd).where(
        LLMCall.created_at >= cutoff
    )
    if transport_kind is not None:
        stmt = stmt.where(LLMCall.transport_kind == transport_kind)

    rows = db.execute(stmt).all()

    total_calls = len(rows)
    error_count = sum(1 for _, success, _ in rows if not success)
    error_rate = error_count / total_calls if total_calls else 0.0
    total_cost = sum(cost for _, _, cost in rows if cost is not None)

    latencies = sorted(latency for latency, _, _ in rows)
    p50 = _percentile(latencies, 50)
    p95 = _percentile(latencies, 95)

    return LLMCallStats(
        window_days=window_days,
        total_calls=total_calls,
        error_count=error_count,
        error_rate=error_rate,
        latency_p50_ms=p50,
        latency_p95_ms=p95,
        total_cost_usd=total_cost,
    )


def _percentile(sorted_values: list[int], pct: int) -> int | None:
    """Nearest-rank percentile of a pre-sorted list, or None if empty.

    Uses the nearest-rank method: the value at the ceil(pct/100 * n)
    position. Simple, dependency-free, and adequate for a latency
    summary. Returns None for an empty list so the caller surfaces
    "no data" rather than a meaningless zero.
    """
    if not sorted_values:
        return None
    n = len(sorted_values)
    # Nearest-rank: rank index is ceil(pct/100 * n) - 1, clamped.
    rank = -(-pct * n // 100) - 1  # ceil division via negation
    rank = max(0, min(rank, n - 1))
    return sorted_values[rank]
