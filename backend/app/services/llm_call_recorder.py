"""Records one observability row per LLM transport round-trip.

The instrumented transport wrapper calls a recorder after every
round-trip with the call's timing, size, token, and outcome data.
Two implementations exist: one writes a row, one does nothing.

The recorder owns its database access rather than borrowing the
caller's session. A turn that calls the LLM and then fails rolls
its own session back. The call still happened, so its record must
survive that rollback. The writing recorder opens a short session
from SessionLocal, commits, and closes, independent of any turn
transaction. This is the deliberate divergence from error_log,
which shares the service session on purpose.

The no-op recorder is the test and eval default. Most tests
exercise transport mechanism against a fake and have no interest in
observability rows, injecting the no-op keeps them from touching
SessionLocal (which binds to the process engine, not a test's
in-memory one) and from polluting any database. Tests that assert
on observability inject the writing recorder bound to their own
session factory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from app.models import LLMCall

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession
    from sqlalchemy.orm import sessionmaker

    from app.models import TransportKind


@dataclass(frozen=True)
class LLMCallData:
    """Everything the wrapper observed about one round-trip.

    A plain data carrier from the wrapper to the recorder. Holds
    what the wrapper can see at the Protocol boundary: which
    transport and method, sizes and timing it measured, tokens and
    model the transport reported, and the outcome. Nothing here
    needs a database to compute.
    """

    trace_id: str
    transport_kind: TransportKind
    method: str
    latency_ms: int
    prompt_chars: int
    response_chars: int
    success: bool
    model: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    error: str | None = None


class LLMCallRecorder(Protocol):
    """Records one LLM round-trip. Implementations decide where to."""

    def record(self, data: LLMCallData) -> None:
        """Persist or discard one call's observability data.

        Must not raise: a recorder failure cannot be allowed to break
        the LLM round-trip it is observing. Implementations swallow
        their own errors.
        """
        ...


class NoOpRecorder:
    """Discards every call. The test and eval default."""

    def record(self, data: LLMCallData) -> None:
        """Do nothing. Observability is off for this transport."""


class WritingRecorder:
    """Writes one llm_call row per call, in its own short session.

    Holds a session factory rather than a live session so each write
    is independent of any turn transaction. The factory is
    SessionLocal in production and a test's own factory in
    observability tests.
    """

    def __init__(self, session_factory: sessionmaker[DbSession]) -> None:
        self._session_factory = session_factory

    def record(self, data: LLMCallData) -> None:
        """Write one row and commit, swallowing any failure.

        A failure to record observability data must never propagate
        into the round-trip being observed. The session is rolled
        back and closed on any error so a broken write leaves no
        half-open transaction behind.
        """
        session = self._session_factory()
        try:
            session.add(
                LLMCall(
                    trace_id=data.trace_id,
                    session_id=None,
                    transport_kind=data.transport_kind,
                    method=data.method,
                    model=data.model,
                    latency_ms=data.latency_ms,
                    prompt_chars=data.prompt_chars,
                    response_chars=data.response_chars,
                    prompt_tokens=data.prompt_tokens,
                    completion_tokens=data.completion_tokens,
                    cost_usd=data.cost_usd,
                    success=data.success,
                    error=data.error,
                )
            )
            session.commit()
        except Exception:
            session.rollback()
        finally:
            session.close()
