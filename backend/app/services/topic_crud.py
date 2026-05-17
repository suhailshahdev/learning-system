"""Topic CRUD primitives.

Functions that create, read, or update Topic rows. Lives outside
session_service so both session_service and tool handlers can use
the same upsert logic without circular imports.

Topic.domain is denormalized from the first segment of Topic.path.
Functions here preserve that invariant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.models import Topic, TopicStatus

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession


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
