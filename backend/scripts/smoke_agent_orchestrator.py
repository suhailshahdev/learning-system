"""Smoke test for the agent orchestrator's transaction contract.

This smoke exists to prove one invariant against real Postgres: a plan
that fails partway through its mutations leaves no partial writes,
and the error record survives the rollback. SQLite unit tests check
the orchestrator's logic but cannot be trusted on the rollback
semantics that matter in production, so this runs against the
Postgres container from docker-compose.

The mutate action is mark-for-revision, which is strict: it requires
the target topic to exist. So each scenario seeds its topics (status
LEARNED) before running the plan, and asserts on the status flip to
NEEDS_REVISION rather than on row creation.

Two scenarios:

  A. Approved plan with valid mutations commits atomically: both
     seeded topics are NEEDS_REVISION after the run.
  B. Approved plan whose second mutate step targets a nonexistent
     topic raises TopicNotFoundError inside the mutate pass, rolls
     the whole transaction back (the first topic's status reverts to
     LEARNED) AND writes an error_log row (it survived the rollback
     because it is on an independent session). This is the two-session
     split from the orchestrator's contract, proven end to end.

Run with the Postgres container up and migrated to head:

    docker compose up -d
    cd backend && uv run python -m scripts.smoke_agent_orchestrator

The script is idempotent: it uses throwaway topic paths under a
recognizable prefix and deletes everything it wrote (topics and
error rows) in a finally block, so a re-run starts clean and a
mid-run failure still cleans up.
"""

from __future__ import annotations

import asyncio
import sys

from app.models import ErrorLog, Topic, TopicStatus
from app.schemas.agent_plan import MarkForRevisionStep, Plan
from app.schemas.tools import MarkForRevisionInput
from app.services.agent_error_recorder import WritingAgentErrorRecorder
from app.services.agent_orchestrator import AgentOrchestratorError, run_plan
from app.services.topic_crud import get_or_create_topic
from sqlalchemy import delete, select

# Recognizable prefix so cleanup can find exactly what this smoke
# wrote without touching real topics. Every topic path here lives
# under this marker.
_SMOKE_PREFIX = "SmokeAgent"
_TOPIC_A = f"{_SMOKE_PREFIX} > Atomic > StepOne"
_TOPIC_B = f"{_SMOKE_PREFIX} > Atomic > StepTwo"
_TOPIC_ROLLBACK = f"{_SMOKE_PREFIX} > Rollback > StepOne"
# A path under the smoke prefix that is deliberately never seeded, so
# marking it raises TopicNotFoundError. Under the prefix so cleanup
# would catch it even if a bug created it.
_TOPIC_MISSING = f"{_SMOKE_PREFIX} > Rollback > NeverSeeded"

_ERROR_KIND = "agent.plan.mutate_failed"


def _require_postgres() -> None:
    """Abort unless the configured database is Postgres.

    The whole point is real rollback semantics. Running this against
    SQLite would pass without testing what it claims to test, so
    refuse rather than give a false green.
    """
    from app.core.config import get_settings  # noqa: PLC0415

    url = get_settings().database_url
    if not url.startswith("postgresql"):
        print(f"FAIL: smoke requires Postgres, got {url!r}.")
        print("Start the container and set the database URL to Postgres.")
        sys.exit(1)


def _seed_topics(paths: list[str]) -> None:
    """Create each path as a committed topic with status LEARNED.

    The strict mutate action requires existing topics. Seeding on its
    own short session keeps the seed independent of the orchestrator's
    transaction, so a scenario rollback cannot revert the seed itself,
    only the status change the plan staged.
    """
    from app.core.db import SessionLocal  # noqa: PLC0415

    session = SessionLocal()
    try:
        for path in paths:
            topic = get_or_create_topic(session, path)
            topic.status = TopicStatus.LEARNED
        session.commit()
    finally:
        session.close()


def _mark_step(path: str) -> MarkForRevisionStep:
    """A mutate step marking the given path for revision."""
    return MarkForRevisionStep(args=MarkForRevisionInput(path=path))


def _status_of(path: str) -> TopicStatus | None:
    """Read one topic's status on a fresh session, or None if absent."""
    from app.core.db import SessionLocal  # noqa: PLC0415

    session = SessionLocal()
    try:
        return session.execute(select(Topic.status).where(Topic.path == path)).scalar_one_or_none()
    finally:
        session.close()


async def _scenario_atomic_commit() -> list[str]:
    """Approved plan with two valid mutations commits both status flips.

    Returns the failures it found (empty list means pass) so the
    caller can aggregate across scenarios before deciding exit code.
    """
    from app.core.db import SessionLocal  # noqa: PLC0415

    failures: list[str] = []
    _seed_topics([_TOPIC_A, _TOPIC_B])
    plan = Plan(steps=[_mark_step(_TOPIC_A), _mark_step(_TOPIC_B)])

    db = SessionLocal()
    recorder = WritingAgentErrorRecorder(SessionLocal)
    try:
        await run_plan(db=db, recorder=recorder, plan=plan, approve=True)
    finally:
        db.close()

    if _status_of(_TOPIC_A) is not TopicStatus.NEEDS_REVISION:
        failures.append(f"A: {_TOPIC_A!r} not NEEDS_REVISION after approved commit.")
    if _status_of(_TOPIC_B) is not TopicStatus.NEEDS_REVISION:
        failures.append(f"A: {_TOPIC_B!r} not NEEDS_REVISION after approved commit.")

    if not failures:
        print("PASS A: approved plan committed both mutations atomically.")
    return failures


async def _scenario_rollback_with_surviving_error() -> list[str]:
    """Failed plan reverts its mutation but persists the error row.

    The plan marks a seeded topic, then marks a topic that was never
    seeded. The second step raises TopicNotFoundError inside the
    mutate pass. The orchestrator must revert the first topic's status
    and record the failure on its independent session.
    """
    from app.core.db import SessionLocal  # noqa: PLC0415

    failures: list[str] = []
    _seed_topics([_TOPIC_ROLLBACK])
    plan = Plan(
        steps=[
            _mark_step(_TOPIC_ROLLBACK),
            _mark_step(_TOPIC_MISSING),
        ]
    )

    db = SessionLocal()
    recorder = WritingAgentErrorRecorder(SessionLocal)
    raised = False
    try:
        await run_plan(db=db, recorder=recorder, plan=plan, approve=True)
    except AgentOrchestratorError:
        raised = True
    finally:
        db.close()

    if not raised:
        failures.append("B: expected AgentOrchestratorError, none raised.")

    if _status_of(_TOPIC_ROLLBACK) is not TopicStatus.LEARNED:
        failures.append(
            f"B: {_TOPIC_ROLLBACK!r} not reverted to LEARNED; rollback did not discard the change."
        )

    error_present = _error_row_present()
    if not error_present:
        failures.append("B: no error_log row after failure; error did not survive the rollback.")

    if not failures:
        print("PASS B: failed plan reverted the mutation and the error row survived.")
    return failures


def _error_row_present() -> bool:
    """True if this smoke's mutate-failed error row exists.

    Scoped by the kind the orchestrator writes plus the never-seeded
    path the failure message embeds, so it matches only this smoke's
    error row and not a real agent error from other runs.
    """
    from app.core.db import SessionLocal  # noqa: PLC0415

    session = SessionLocal()
    try:
        return (
            session.execute(
                select(ErrorLog.id).where(
                    ErrorLog.kind == _ERROR_KIND,
                    ErrorLog.message.like(f"%{_TOPIC_MISSING}%"),
                )
            ).first()
            is not None
        )
    finally:
        session.close()


def _cleanup() -> None:
    """Delete everything the smoke wrote: topics and the error row.

    Runs regardless of pass or fail so a re-run starts clean. Matches
    topics by the smoke prefix and the error row by the kind the
    orchestrator writes plus the never-seeded path marker in the
    message, so it cannot delete a real agent error from other runs.
    """
    from app.core.db import SessionLocal  # noqa: PLC0415

    session = SessionLocal()
    try:
        session.execute(delete(Topic).where(Topic.path.like(f"{_SMOKE_PREFIX} > %")))
        session.execute(
            delete(ErrorLog).where(
                ErrorLog.kind == _ERROR_KIND,
                ErrorLog.message.like(f"%{_TOPIC_MISSING}%"),
            )
        )
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"WARN: cleanup failed: {e}")
    finally:
        session.close()


async def _main() -> int:
    _require_postgres()
    failures: list[str] = []
    try:
        failures += await _scenario_atomic_commit()
        failures += await _scenario_rollback_with_surviving_error()
    finally:
        _cleanup()

    if failures:
        print("\nSMOKE FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nSMOKE PASSED: orchestrator transaction contract holds on Postgres.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main()))
