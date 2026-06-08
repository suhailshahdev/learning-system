"""Tests for the agent orchestrator's logic and the commit-free guarantee.

These cover what runs in the fast suite on in-memory SQLite: the
read/mutate phase split, the approve gate, and the flush-not-commit
guarantee of the agent action tool. The no-op recorder is injected so
nothing touches SessionLocal.

What these deliberately do not cover is the cross-session rollback-
with-surviving-error invariant. That depends on real Postgres
rollback semantics and is owned by smoke_agent_orchestrator.py. The
split is intentional: unit tests own logic and the commit-free
guarantee, the smoke owns the real-DB transaction contract.

The plan step vocabulary is a closed discriminated union, so a plan
with an unknown tool cannot be constructed: Pydantic rejects it at
validation. The orchestrator's unknown-tool dispatch arms are
therefore unreachable from the public Plan API and are not tested
here. They are a backstop for a future union member added without a
dispatch arm, which mypy catches at the dispatch site. The natural
mutate-failure mode is now a step naming a topic that does not exist,
which raises TopicNotFoundError inside the mutate pass: a real
runtime condition, exercised below.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from app.models import Topic, TopicStatus
from app.schemas.agent_plan import (
    GetWeakTopicsStep,
    MarkForRevisionStep,
    Plan,
)
from app.schemas.tools import GetWeakTopicsInput, MarkForRevisionInput
from app.services.agent_error_recorder import NoOpAgentErrorRecorder
from app.services.agent_orchestrator import AgentOrchestratorError, run_plan
from app.services.agent_tools import stage_mark_for_revision
from app.services.topic_crud import get_or_create_topic

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession


def _seed_topic(db: DbSession, path: str, status: TopicStatus = TopicStatus.LEARNED) -> None:
    """Create a committed topic at path with a known starting status.

    The strict mutate tool requires the topic to exist, so mutate
    tests seed targets first. Committing makes them durable state the
    orchestrator reads, not rows staged in the test's own session.
    """
    topic = get_or_create_topic(db, path)
    topic.status = status
    db.commit()


def _status_of(db: DbSession, path: str) -> TopicStatus:
    """Read one topic's current status, asserting it exists."""
    return db.query(Topic).filter(Topic.path == path).one().status


def _weak_step() -> GetWeakTopicsStep:
    """A read step with valid get_weak_topics args."""
    return GetWeakTopicsStep(args=GetWeakTopicsInput(min_attempts=1))


def _mark_step(path: str) -> MarkForRevisionStep:
    """A mutate step marking the given path for revision."""
    return MarkForRevisionStep(args=MarkForRevisionInput(path=path))


# --- The commit-free guarantee (the agent action tool) ---


def test_stage_mark_for_revision_flushes_but_does_not_commit(db: DbSession) -> None:
    """The wrapper stages the status change, a rollback reverts it.

    This is the agent-path guarantee at the tool layer: the tool
    flushes through the core so later steps see the change, but does
    not commit, so the orchestrator owns the transaction. If it
    wrongly committed, the rollback below would not restore the prior
    status. The core's own behavior is tested in test_topic_crud.py,
    this pins that the wrapper preserves it.
    """
    path = "Python > Async > Coroutines"
    _seed_topic(db, path, TopicStatus.LEARNED)

    returned = stage_mark_for_revision(db, path=path)

    assert returned == path
    assert _status_of(db, path) is TopicStatus.NEEDS_REVISION  # visible after flush

    db.rollback()
    assert _status_of(db, path) is TopicStatus.LEARNED  # discarded, so never committed


# --- The read/mutate phase split and the approve gate ---


async def test_read_pass_returns_evidence_and_mutates_nothing(db: DbSession) -> None:
    """approve=False runs reads, returns evidence, and writes no mutations.

    Even though the plan carries a mutate step against a real topic,
    the gate stops before the mutate pass, so the topic's status must
    be unchanged afterwards.
    """
    path = "Python > Async > EventLoop"
    _seed_topic(db, path, TopicStatus.LEARNED)
    plan = Plan(steps=[_weak_step(), _mark_step(path)])

    evidence = await run_plan(db=db, recorder=NoOpAgentErrorRecorder(), plan=plan, approve=False)

    assert len(evidence) == 1
    assert evidence[0].tool == "get_weak_topics"
    assert _status_of(db, path) is TopicStatus.LEARNED


async def test_approved_plan_commits_mutations(db: DbSession) -> None:
    """approve=True runs the mutate pass and the status changes persist.

    SQLite-level happy path: checks the orchestrator's logic (reads
    then mutates, commits once), not the rollback semantics the smoke
    owns. Both seeded topics flip to needs_revision and a fresh query
    confirms the commit.
    """
    path_a = "Python > Async > Tasks"
    path_b = "Python > Async > Gather"
    _seed_topic(db, path_a, TopicStatus.LEARNED)
    _seed_topic(db, path_b, TopicStatus.LEARNED)
    plan = Plan(steps=[_mark_step(path_a), _mark_step(path_b)])

    await run_plan(db=db, recorder=NoOpAgentErrorRecorder(), plan=plan, approve=True)

    assert _status_of(db, path_a) is TopicStatus.NEEDS_REVISION
    assert _status_of(db, path_b) is TopicStatus.NEEDS_REVISION


async def test_read_pass_skips_mutate_steps(db: DbSession) -> None:
    """The read pass ignores mutate steps, only reads produce evidence.

    A plan with one mutate and one read, run with approve=False,
    yields exactly one evidence entry (the read), never the mutate.
    Step order is mutate-first to prove the read pass filters by kind
    rather than by position.
    """
    path = "Python > Async > Cancellation"
    _seed_topic(db, path, TopicStatus.LEARNED)
    plan = Plan(steps=[_mark_step(path), _weak_step()])

    evidence = await run_plan(db=db, recorder=NoOpAgentErrorRecorder(), plan=plan, approve=False)

    assert len(evidence) == 1
    assert evidence[0].tool == "get_weak_topics"


# --- Mutate-pass failure rolls back atomically ---


async def test_mutate_failure_rolls_back_prior_mutations(db: DbSession) -> None:
    """A failing mutate step discards a prior successful one in the same plan.

    The plan marks a real topic, then marks a nonexistent one. The
    second step raises TopicNotFoundError inside the mutate pass. The
    orchestrator rolls the whole transaction back before raising, so
    the first topic's status must revert to its committed value. This
    is the same control flow the smoke proves on Postgres, here
    confirming the logic and the rollback call on SQLite.
    """
    real_path = "Python > Async > Semaphore"
    _seed_topic(db, real_path, TopicStatus.LEARNED)
    plan = Plan(
        steps=[
            _mark_step(real_path),
            _mark_step("Nonexistent > Topic > Path"),
        ]
    )

    with pytest.raises(AgentOrchestratorError):
        await run_plan(db=db, recorder=NoOpAgentErrorRecorder(), plan=plan, approve=True)

    assert _status_of(db, real_path) is TopicStatus.LEARNED
