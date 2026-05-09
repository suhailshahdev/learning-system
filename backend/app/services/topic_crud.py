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
