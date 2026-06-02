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

from app.services.topic_crud import get_or_create_topic

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession


def stage_topic_upsert(db: DbSession, *, path: str) -> str:
    """Stage a topic upsert without committing. Returns the topic path.

    Calls the commit-free get_or_create_topic core, which flushes so
    the row is visible to later steps in the same transaction but
    does not commit. The orchestrator commits once after all mutate
    steps succeed, or rolls back the lot on any failure.

    This is the agent-path counterpart to the create_or_update_topic
    tool handler, minus the handler's own commit.
    """
    topic = get_or_create_topic(db, path)
    return topic.path
