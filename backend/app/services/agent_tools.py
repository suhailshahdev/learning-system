"""Mutating action tools for the agent orchestrator.

These differ from the teaching-loop tool handlers in one way that
matters: they flush, they do not commit. The orchestrator owns the
transaction so an approved multi-step plan applies atomically or not
at all. The teaching-loop handlers commit per call on purpose
(one mutation per turn, durability wanted). The agent wants the
opposite guarantee, so it has its own thin action layer that calls
the commit-free CRUD cores and leaves the commit to the caller.

Ships one action: stage a topic upsert. It callsget_or_create_topic,
the same commit-free primitive the teaching tool handler wraps, and
returns the resulting path. This will grow into the full action
executor surface in a later step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.services.topic_crud import mark_topic_for_revision

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession


def stage_mark_for_revision(db: DbSession, *, path: str) -> str:
    """Stage a mark-for-revision without committing. Returns the topic path.

    Calls the commit-free mark_topic_for_revision core, which sets the
    topic's status to needs_revision and flushes so the change is
    visible to later steps in the same transaction but does not
    commit. The orchestrator commits once after all mutate steps
    succeed, or rolls the whole plan back on any failure.

    Strict by way of the core: a path that does not resolve to an
    existing topic raises TopicNotFoundError, which fails the mutate
    pass and rolls the plan back. The planner's groundedness guard
    rejects ungrounded targets before approval, so this raise is a
    backstop, not the normal path.

    This is the agent-path counterpart to a teaching tool handler,
    minus the handler's own commit (D488).
    """
    topic = mark_topic_for_revision(db, path)
    return topic.path
