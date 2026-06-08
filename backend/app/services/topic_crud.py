"""Topic CRUD primitives.

Functions that create, read, or update Topic rows. Lives outside
session_service so both session_service and tool handlers can use
the same upsert logic without circular imports.

Topic.domain is denormalized from the first segment of Topic.path.
Functions here preserve that invariant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import select

from app.models import Topic, TopicStatus

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession


class TopicNotFoundError(Exception):
    """A topic lookup that required an existing row found none.

    Raised by the strict primitives (mark_topic_for_revision) when
    the path does not resolve to a topic. Distinct from the create-
    on-miss primitives, which never raise on absence. The agent
    planner maps this to a groundedness failure. It should not
    normally reach a user, because the groundedness guard rejects
    ungrounded targets before the mutate pass runs.
    """

    def __init__(self, path: str) -> None:
        super().__init__(f"Topic not found: {path!r}.")
        self.path = path


def get_or_create_topic(db: DbSession, path: str) -> Topic:
    """Find a topic by path or create one.

    Domain is denormalized from the first path segment per the
    Topic model's documented invariant. Used by session_service at
    session start and by the create_or_update_topic tool handler.

    The caller is responsible for committing. This function flushes
    so the new row's id is available for FK references in the same
    transaction.
    """
    existing = db.query(Topic).filter(Topic.path == path).one_or_none()
    if existing is not None:
        return existing

    domain = path.split(" > ", 1)[0]
    name = path.rsplit(" > ", 1)[-1]

    topic = Topic(
        path=path,
        domain=domain,
        name=name,
        status=TopicStatus.IN_PROGRESS,
    )
    db.add(topic)
    db.flush()
    return topic


def get_or_create_topic_with_ancestors(db: DbSession, path: str) -> Topic:
    """Find or create a topic and every missing ancestor in its path.

    Walks the path segments left-to-right. For each cumulative prefix,
    calls get_or_create_topic and wires parent_id from the prior step.
    Returns the leaf topic.

    Ancestors are created with minimal metadata: status defaults to
    IN_PROGRESS, difficulty stays NULL, prerequisites stay empty.
    The caller is expected to set metadata on the leaf separately if
    needed.

    Used by the create_or_update_topic tool handler so the LLM can
    declare a deep path or a parent_path without first calling the
    handler once per ancestor.
    """
    segments = path.split(" > ")
    parent: Topic | None = None
    topic: Topic | None = None
    for i in range(len(segments)):
        prefix = " > ".join(segments[: i + 1])
        topic = get_or_create_topic(db, prefix)
        if topic.parent_id is None and parent is not None:
            topic.parent_id = parent.id
            db.flush()
        parent = topic

    # The loop runs at least once because split(" > ") on a non-empty
    # string yields at least one element. Assert documents that
    # invariant for the type checker.
    assert topic is not None
    return topic


def mark_topic_for_revision(db: DbSession, path: str) -> Topic:
    """Set an existing topic's status to needs_revision. Returns the topic.

    Strict: the topic must already exist. Marking a nonexistent topic
    for revision is meaningless, so a missing path raises
    TopicNotFoundError rather than creating a row. This is the
    deliberate divergence from get_or_create_topic, which creates on
    miss for the teaching loop. Here the user is revising a known weak
    topic, not introducing a new one.

    The caller commits. This function flushes so the status change is
    visible to later steps in the same transaction. The agent
    orchestrator owns the commit so an approved multi-step plan
    applies atomically. stage_mark_for_revision is the commit-free
    wrapper the orchestrator dispatches to.
    """
    topic = db.execute(select(Topic).where(Topic.path == path)).scalar_one_or_none()
    if topic is None:
        raise TopicNotFoundError(path)
    topic.status = TopicStatus.NEEDS_REVISION
    db.flush()
    return topic
