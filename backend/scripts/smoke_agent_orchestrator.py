"""Smoke test for the agent orchestrator's transaction contract.

This smoke exists to prove one invariant against real Postgres: a plan
that fails partway through its mutations leaves no partial writes,
and the error record survives the rollback. SQLite unit tests check
the orchestrator's logic but cannot be trusted on the rollback
semantics that matter in production, so this runs against the
Postgres container from docker-compose.

Two scenarios:

  A. Approved plan with valid mutations commits atomically: both
     staged topics exist after the run.
  B. Approved plan whose second mutate step fails rolls the whole
     transaction back (the first step's topic is absent) AND writes
     an error_log row (it survived the rollback because it is on an
     independent session). This is the two-session split from the
     orchestrator's contract, proven end to end.

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

from app.models import ErrorLog, Topic
from app.schemas.agent_plan import Plan, PlanStep
from app.services.agent_error_recorder import WritingAgentErrorRecorder
from app.services.agent_orchestrator import AgentOrchestratorError, run_plan
from sqlalchemy import delete, select

# Recognizable prefix so cleanup can find exactly what this smoke
# wrote without touching real topics. Every topic path and the error
# context here lives under this marker.
_SMOKE_PREFIX = "SmokeAgent"
_TOPIC_A = f"{_SMOKE_PREFIX} > Atomic > StepOne"
_TOPIC_B = f"{_SMOKE_PREFIX} > Atomic > StepTwo"
_TOPIC_ROLLBACK = f"{_SMOKE_PREFIX} > Rollback > StepOne"

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


async def _scenario_atomic_commit() -> list[str]:
    """Approved plan with two valid mutations commits both.

    Returns the failures it found (empty list means pass) so the
    caller can aggregate across scenarios before deciding exit code.
    """
    from app.core.db import SessionLocal  # noqa: PLC0415

    failures: list[str] = []
    plan = Plan(
        steps=[
            PlanStep(kind="read", tool="get_weak_topics", args={"min_attempts": 1}),
            PlanStep(kind="mutate", tool="stage_topic_upsert", args={"path": _TOPIC_A}),
            PlanStep(kind="mutate", tool="stage_topic_upsert", args={"path": _TOPIC_B}),
        ]
    )

    db = SessionLocal()
    recorder = WritingAgentErrorRecorder(SessionLocal)
    try:
        evidence = await run_plan(db=db, recorder=recorder, plan=plan, approve=True)
    finally:
        db.close()

    if not evidence:
        failures.append("A: expected evidence from the read pass, got none.")

    # Fresh session to read committed state, not the orchestrator's.
    check = SessionLocal()
    try:
        present = set(
            check.execute(select(Topic.path).where(Topic.path.in_([_TOPIC_A, _TOPIC_B])))
            .scalars()
            .all()
        )
    finally:
        check.close()

    if _TOPIC_A not in present:
        failures.append(f"A: {_TOPIC_A!r} missing after approved commit.")
    if _TOPIC_B not in present:
        failures.append(f"A: {_TOPIC_B!r} missing after approved commit.")

    if not failures:
        print("PASS A: approved plan committed both mutations atomically.")
    return failures


async def _scenario_rollback_with_surviving_error() -> list[str]:
    """Failed plan rolls back its mutation but persists the error row.

    The plan stages one valid topic, then hits an unknown mutate tool
    that raises inside the mutate pass. The orchestrator must roll the
    staged topic back and record the failure on its independent
    session.
    """
    from app.core.db import SessionLocal  # noqa: PLC0415

    failures: list[str] = []
    plan = Plan(
        steps=[
            PlanStep(kind="mutate", tool="stage_topic_upsert", args={"path": _TOPIC_ROLLBACK}),
            PlanStep(kind="mutate", tool="does_not_exist", args={}),
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

    check = SessionLocal()
    try:
        topic_present = (
            check.execute(
                select(Topic.path).where(Topic.path == _TOPIC_ROLLBACK)
            ).scalar_one_or_none()
            is not None
        )
        error_present = (
            check.execute(select(ErrorLog.id).where(ErrorLog.kind == _ERROR_KIND)).first()
            is not None
        )
    finally:
        check.close()

    if topic_present:
        failures.append(
            f"B: {_TOPIC_ROLLBACK!r} present after failure; rollback did not discard it."
        )
    if not error_present:
        failures.append("B: no error_log row after failure; error did not survive the rollback.")

    if not failures:
        print("PASS B: failed plan rolled back the mutation and the error row survived.")
    return failures


def _cleanup() -> None:
    """Delete everything the smoke wrote: topics and error rows.

    Runs regardless of pass or fail so a re-run starts clean.
    Matches topics by the smoke prefix and error rows by the kind
    the orchestrator writes plus a path marker in the message, so it
    cannot delete a real agent error from other runs.
    """
    from app.core.db import SessionLocal  # noqa: PLC0415

    session = SessionLocal()
    try:
        session.execute(delete(Topic).where(Topic.path.like(f"{_SMOKE_PREFIX} > %")))
        # The rollback scenario writes one error row per run. Scope the
        # delete by the sentinel tool name the failure message embeds:
        # nothing real uses the tool name "does_not_exist", so matching
        # the message keeps cleanup to this smoke's writes without
        # reaching into the JSON context column (which is the generic
        # JSON type and has no portable indexed-text accessor).
        session.execute(
            delete(ErrorLog).where(
                ErrorLog.kind == _ERROR_KIND,
                ErrorLog.message.like("%does_not_exist%"),
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
