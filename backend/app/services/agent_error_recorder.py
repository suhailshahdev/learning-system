"""Records one error row per failed agent-orchestrator run.

The orchestrator stages mutations across multiple steps and applies
them in one transaction. When a step fails, the orchestrator rolls
that transaction back so no partial plan persists. The error record
must survive that rollback: it describes a failure the rolled-back
session can no longer hold.

This is the same situation the LLM-call recorder solves, and the
same divergence from error_log's shared-session pattern. error_log
relies on the service not having written rows before the failure
point. The orchestrator violates that by design, since staged
mutations sit in the session across steps. So the agent path logs
on an independent session, mirroring the LLM-call recorder rather
than session_service's _log_service_error.

A Protocol with writing and no-op implementations, matching the
LLM-call recorder, so unit tests on the in-memory engine inject the
no-op and never touch SessionLocal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from app.models import ErrorLog

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession
    from sqlalchemy.orm import sessionmaker


@dataclass(frozen=True)
class AgentErrorData:
    """What the orchestrator observed about one failed run.

    A plain carrier from the orchestrator to the recorder. kind is a
    snake_case dot-form string matching the error_log convention.
    context is free-form JSON detail: which step failed, the tool,
    the underlying message.
    """

    kind: str
    message: str
    context: dict[str, Any]


class AgentErrorRecorder(Protocol):
    """Records one agent-run failure. Implementations decide where to."""

    def record(self, data: AgentErrorData) -> None:
        """Persist or discard one failure's data.

        Must not raise: a recorder failure cannot mask the original
        error the orchestrator is already raising. Implementations
        swallow their own errors.
        """
        ...


class NoOpAgentErrorRecorder:
    """Discards every record. The unit-test default."""

    def record(self, data: AgentErrorData) -> None:
        """Do nothing."""


class WritingAgentErrorRecorder:
    """Writes one error_log row per failure, in its own short session.

    Holds a session factory, not a live session, so each write is
    independent of the orchestrator's transaction. The factory is
    SessionLocal in production and a test's own factory in tests
    that assert on the error row.

    session_id is always None: the assistant has no session row, so
    agent errors are not correlated to a learning session.
    """

    def __init__(self, session_factory: sessionmaker[DbSession]) -> None:
        self._session_factory = session_factory

    def record(self, data: AgentErrorData) -> None:
        """Write one row and commit, swallowing any failure.

        The original orchestrator error matters more than a failure
        to log it. The session is rolled back and closed on any error
        so a broken write leaves no half-open transaction.
        """
        session = self._session_factory()
        try:
            session.add(
                ErrorLog(
                    session_id=None,
                    kind=data.kind,
                    message=data.message,
                    context=data.context,
                    trace_id=None,
                )
            )
            session.commit()
        except Exception:
            session.rollback()
        finally:
            session.close()
