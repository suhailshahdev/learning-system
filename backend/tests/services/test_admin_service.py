"""Tests for the LLM-call observability read service.

Covers the pure percentile helper, the row-list filters and cap,
and the stats aggregate including the window boundary. Each test
seeds llm_call rows and exercises one behavior. Helpers are
private to this file, same pattern as the browse-service tests.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from app.models import LLMCall, TransportKind
from app.services.admin_service import (
    LLM_CALL_LIMIT,
    _percentile,
    list_llm_calls,
    llm_call_stats,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession


def _make_call(
    db: DbSession,
    *,
    transport_kind: TransportKind = TransportKind.DEEPSEEK,
    method: str = "send",
    latency_ms: int = 100,
    success: bool = True,
    cost_usd: float | None = None,
    created_at: datetime | None = None,
    trace_id: str = "trace-fixed",
) -> LLMCall:
    """Seed one llm_call row with sensible defaults."""
    call = LLMCall(
        trace_id=trace_id,
        session_id=None,
        transport_kind=transport_kind,
        method=method,
        model=None,
        latency_ms=latency_ms,
        prompt_chars=10,
        response_chars=20,
        prompt_tokens=None,
        completion_tokens=None,
        cost_usd=cost_usd,
        success=success,
        error=None if success else "boom",
        created_at=created_at or datetime(2026, 5, 30, tzinfo=UTC),
    )
    db.add(call)
    db.flush()
    return call


# --- _percentile: pure function, exact cases ---


def test_percentile_empty_returns_none() -> None:
    """An empty list has no percentile."""
    assert _percentile([], 50) is None
    assert _percentile([], 95) is None


def test_percentile_single_value() -> None:
    """A single value is its own percentile at any pct."""
    assert _percentile([42], 50) == 42
    assert _percentile([42], 95) == 42


def test_percentile_p50_and_p95_known_list() -> None:
    """Nearest-rank p50 and p95 on a hand-computed list.

    For [10, 20, 30, 40, 50] (n=5):
      p50: ceil(0.50 * 5) = 3, index 2 -> 30
      p95: ceil(0.95 * 5) = 5, index 4 -> 50
    """
    values = [10, 20, 30, 40, 50]
    assert _percentile(values, 50) == 30
    assert _percentile(values, 95) == 50


def test_percentile_ceil_rounding_boundary() -> None:
    """p95 of a 10-element list rounds up, not down.

    For n=10, p95: ceil(0.95 * 10) = 10, index 9 -> the largest.
    A floor-based implementation would land on index 8 and return
    the wrong value. This is the off-by-one guard.
    """
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert _percentile(values, 95) == 10
    # p50 of n=10: ceil(0.50 * 10) = 5, index 4 -> 5.
    assert _percentile(values, 50) == 5


# --- list_llm_calls: filters, ordering, cap ---


async def test_list_returns_empty_when_no_calls(db: DbSession) -> None:
    """No rows seeded yields empty list and limit_reached=False."""
    response = list_llm_calls(db=db)

    assert response.rows == []
    assert response.limit_reached is False


async def test_list_orders_by_created_at_desc(db: DbSession) -> None:
    """Rows come back newest first."""
    older = _make_call(db, created_at=datetime(2026, 5, 1, tzinfo=UTC))
    newer = _make_call(db, created_at=datetime(2026, 5, 3, tzinfo=UTC))
    middle = _make_call(db, created_at=datetime(2026, 5, 2, tzinfo=UTC))
    db.commit()

    response = list_llm_calls(db=db)

    assert [row.id for row in response.rows] == [newer.id, middle.id, older.id]


async def test_list_filters_by_transport_kind(db: DbSession) -> None:
    """transport_kind narrows to one transport."""
    _make_call(db, transport_kind=TransportKind.DEEPSEEK)
    _make_call(db, transport_kind=TransportKind.DEEPSEEK)
    _make_call(db, transport_kind=TransportKind.CLAUDE_PLAYWRIGHT)
    db.commit()

    response = list_llm_calls(db=db, transport_kind=TransportKind.CLAUDE_PLAYWRIGHT)

    assert len(response.rows) == 1
    assert response.rows[0].transport_kind == TransportKind.CLAUDE_PLAYWRIGHT


async def test_list_filters_by_success(db: DbSession) -> None:
    """success=False returns only failures."""
    _make_call(db, success=True)
    _make_call(db, success=False)
    _make_call(db, success=False)
    db.commit()

    response = list_llm_calls(db=db, success=False)

    assert len(response.rows) == 2
    assert all(row.success is False for row in response.rows)
    assert all(row.error == "boom" for row in response.rows)


async def test_list_caps_at_limit_and_signals_reached(db: DbSession) -> None:
    """Seeding LLM_CALL_LIMIT + 1 rows caps and sets limit_reached.

    Falsifying test for the limit+1 trick, mirroring the browse
    service's cap test.
    """
    base = datetime(2026, 5, 1, tzinfo=UTC)
    for i in range(LLM_CALL_LIMIT + 1):
        _make_call(db, created_at=base + timedelta(minutes=i))
    db.commit()

    response = list_llm_calls(db=db)

    assert len(response.rows) == LLM_CALL_LIMIT
    assert response.limit_reached is True


async def test_list_at_exactly_limit_does_not_signal_reached(db: DbSession) -> None:
    """Exactly LLM_CALL_LIMIT rows leaves limit_reached=False."""
    base = datetime(2026, 5, 1, tzinfo=UTC)
    for i in range(LLM_CALL_LIMIT):
        _make_call(db, created_at=base + timedelta(minutes=i))
    db.commit()

    response = list_llm_calls(db=db)

    assert len(response.rows) == LLM_CALL_LIMIT
    assert response.limit_reached is False


# --- llm_call_stats: aggregates and window boundary ---


async def test_stats_empty_window_is_all_zeros(db: DbSession) -> None:
    """An empty window reports zeros and null percentiles, not errors."""
    stats = llm_call_stats(db=db, window_days=7)

    assert stats.total_calls == 0
    assert stats.error_count == 0
    assert stats.error_rate == 0.0
    assert stats.latency_p50_ms is None
    assert stats.latency_p95_ms is None
    assert stats.total_cost_usd == 0.0


async def test_stats_counts_and_error_rate(db: DbSession) -> None:
    """Mixed success/failure yields correct error_count and rate.

    Recent created_at so all rows fall inside the default window.
    """
    now = datetime.now(UTC)
    _make_call(db, success=True, created_at=now)
    _make_call(db, success=True, created_at=now)
    _make_call(db, success=True, created_at=now)
    _make_call(db, success=False, created_at=now)
    db.commit()

    stats = llm_call_stats(db=db, window_days=7)

    assert stats.total_calls == 4
    assert stats.error_count == 1
    assert stats.error_rate == 0.25


async def test_stats_latency_percentiles(db: DbSession) -> None:
    """Percentiles computed over the window's latencies.

    Latencies [10, 20, 30, 40, 50], all in-window:
      p50 -> 30, p95 -> 50 (same nearest-rank math as the unit test).
    """
    now = datetime.now(UTC)
    for latency in (10, 20, 30, 40, 50):
        _make_call(db, latency_ms=latency, created_at=now)
    db.commit()

    stats = llm_call_stats(db=db, window_days=7)

    assert stats.latency_p50_ms == 30
    assert stats.latency_p95_ms == 50


async def test_stats_window_excludes_old_calls(db: DbSession) -> None:
    """A call older than the window is excluded, one inside is counted.

    The boundary check on the created_at >= cutoff comparison.
    Seeds one call 10 days old and one 1 day old, asks for a
    7-day window, expects only the recent one.
    """
    now = datetime.now(UTC)
    _make_call(db, created_at=now - timedelta(days=10))
    _make_call(db, created_at=now - timedelta(days=1))
    db.commit()

    stats = llm_call_stats(db=db, window_days=7)

    assert stats.total_calls == 1


async def test_stats_total_cost_sums_non_null(db: DbSession) -> None:
    """total_cost_usd sums known costs and treats null as zero.

    Two rows with cost, one with null. sum is the two known ones.
    """
    now = datetime.now(UTC)
    _make_call(db, cost_usd=0.01, created_at=now)
    _make_call(db, cost_usd=0.02, created_at=now)
    _make_call(db, cost_usd=None, created_at=now)
    db.commit()

    stats = llm_call_stats(db=db, window_days=7)

    assert stats.total_calls == 3
    assert stats.total_cost_usd == 0.03


async def test_stats_filters_by_transport_kind(db: DbSession) -> None:
    """transport_kind narrows the aggregate to one transport."""
    now = datetime.now(UTC)
    _make_call(db, transport_kind=TransportKind.DEEPSEEK, created_at=now)
    _make_call(db, transport_kind=TransportKind.DEEPSEEK, created_at=now)
    _make_call(db, transport_kind=TransportKind.CLAUDE_PLAYWRIGHT, created_at=now)
    db.commit()

    stats = llm_call_stats(db=db, window_days=7, transport_kind=TransportKind.DEEPSEEK)

    assert stats.total_calls == 2
