"""Tests for the agent orchestrator's logic and the commit-free guarantee.

These cover what runs in the fast suite on in-memory SQLite: the
read/mutate phase split, the approve gate, dispatch guards, and the
flush-not-commit guarantee of the agent action tools. The no-op
recorder is injected so nothing touches SessionLocal.

What these deliberately do not cover is the cross-session rollback-
with-surviving-error invariant. That depends on real Postgres
rollback semantics and is owned by smoke_agent_orchestrator.py. The
split is intentional: unit tests own logic and the commit-free
guarantee, the smoke owns the real-DB transaction contract.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from app.models import Topic
from app.schemas.agent_plan import Plan, PlanStep
from app.services.agent_error_recorder import NoOpAgentErrorRecorder
from app.services.agent_orchestrator import AgentOrchestratorError, run_plan
from app.services.agent_tools import stage_topic_upsert

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession


def _topic_exists(db: DbSession, path: str) -> bool:
    """True if a topic with this path is visible to the session."""
    return db.query(Topic).filter(Topic.path == path).one_or_none() is not None


# --- The commit-free guarantee ---


def test_stage_topic_upsert_flushes_but_does_not_commit(db: DbSession) -> None:
    """stage_topic_upsert makes the row visible, but a rollback discards it.

    This is the agent-path guarantee: the tool stages without
    committing so the orchestrator owns the transaction. If it
    wrongly committed, the rollback below would not discard the row.
    """
    path = "Python > Agent > Staged"
    returned = stage_topic_upsert(db, path=path)

    assert returned == path
    assert _topic_exists(db, path)  # visible after flush

    db.rollback()
    assert not _topic_exists(db, path)  # discarded, so it was never committed


def test_stage_topic_upsert_returns_existing_path(db: DbSession) -> None:
    """Upserting an existing path returns it without creating a duplicate."""
    path = "Python > Agent > Existing"
    stage_topic_upsert(db, path=path)
    db.flush()

    stage_topic_upsert(db, path=path)
    count = db.query(Topic).filter(Topic.path == path).count()

    assert count == 1


# --- The read/mutate phase split and the approve gate ---


async def test_read_pass_returns_evidence_and_mutates_nothing(db: DbSession) -> None:
    """approve=False runs reads, returns evidence, and writes no mutations.

    Even though the plan carries a mutate step, the gate stops before
    the mutate pass, so the staged topic must be absent afterwards.
    """
    mutate_path = "Python > Agent > ShouldNotExist"
    plan = Plan(
        steps=[
            PlanStep(kind="read", tool="get_weak_topics", args={"min_attempts": 1}),
            PlanStep(kind="mutate", tool="stage_topic_upsert", args={"path": mutate_path}),
        ]
    )

    evidence = await run_plan(db=db, recorder=NoOpAgentErrorRecorder(), plan=plan, approve=False)

    assert len(evidence) == 1
    assert evidence[0].tool == "get_weak_topics"
    assert not _topic_exists(db, mutate_path)


async def test_approved_plan_commits_mutations(db: DbSession) -> None:
    """approve=True runs the mutate pass and the staged topics persist.

    SQLite-level happy path: this checks the orchestrator's logic
    (reads then mutates, commits once), not the rollback semantics the
    smoke owns. A fresh query after the run confirms the commit.
    """
    path_a = "Python > Agent > CommittedA"
    path_b = "Python > Agent > CommittedB"
    plan = Plan(
        steps=[
            PlanStep(kind="mutate", tool="stage_topic_upsert", args={"path": path_a}),
            PlanStep(kind="mutate", tool="stage_topic_upsert", args={"path": path_b}),
        ]
    )

    await run_plan(db=db, recorder=NoOpAgentErrorRecorder(), plan=plan, approve=True)

    assert _topic_exists(db, path_a)
    assert _topic_exists(db, path_b)


async def test_read_pass_skips_mutate_steps(db: DbSession) -> None:
    """The read pass ignores mutate steps; only reads produce evidence.

    A plan with one read and one mutate, run with approve=False,
    yields exactly one evidence entry (the read), never the mutate.
    """
    plan = Plan(
        steps=[
            PlanStep(kind="mutate", tool="stage_topic_upsert", args={"path": "Python > X"}),
            PlanStep(kind="read", tool="get_weak_topics", args={"min_attempts": 1}),
        ]
    )

    evidence = await run_plan(db=db, recorder=NoOpAgentErrorRecorder(), plan=plan, approve=False)

    assert len(evidence) == 1
    assert evidence[0].tool == "get_weak_topics"


# --- Dispatch guards ---


async def test_unknown_read_tool_raises(db: DbSession) -> None:
    """An unknown read tool raises AgentOrchestratorError in the read pass."""
    plan = Plan(steps=[PlanStep(kind="read", tool="no_such_read", args={})])

    with pytest.raises(AgentOrchestratorError, match="Unknown read tool"):
        await run_plan(db=db, recorder=NoOpAgentErrorRecorder(), plan=plan, approve=False)


async def test_unknown_mutate_tool_raises_and_rolls_back(db: DbSession) -> None:
    """An unknown mutate tool raises, and a prior staged mutation is discarded.

    The plan stages one valid topic, then hits an unknown mutate tool.
    The orchestrator rolls the transaction back before raising, so the
    first topic must be absent afterwards. This is the same control
    flow the smoke proves on Postgres, here it confirms the logic and
    the rollback call on SQLite.
    """
    staged_path = "Python > Agent > RolledBack"
    plan = Plan(
        steps=[
            PlanStep(kind="mutate", tool="stage_topic_upsert", args={"path": staged_path}),
            PlanStep(kind="mutate", tool="no_such_mutate", args={}),
        ]
    )

    with pytest.raises(AgentOrchestratorError, match="Unknown mutate tool"):
        await run_plan(db=db, recorder=NoOpAgentErrorRecorder(), plan=plan, approve=True)

    assert not _topic_exists(db, staged_path)
