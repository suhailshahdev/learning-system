"""Tests for the strict topic CRUD primitive used by the agent path.

mark_topic_for_revision is the commit-free core the agent
orchestrator dispatches to through stage_mark_for_revision. Unlike
get_or_create_topic, it is strict: a path that does not resolve to an
existing topic raises rather than creating one. These tests cover the
status flip, the flush-not-commit guarantee (visible after flush,
discarded by rollback), and the raise on an absent path.

The create-on-miss primitives (get_or_create_topic,
get_or_create_topic_with_ancestors) are exercised through the tool
handlers and the session service, this file is the direct home for
the strict primitive.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from app.models import Topic, TopicStatus
from app.services.topic_crud import (
    TopicNotFoundError,
    get_or_create_topic,
    mark_topic_for_revision,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession


def _seed_topic(db: DbSession, path: str, status: TopicStatus) -> None:
    """Create a topic at path with an explicit status, committed.

    Uses the create-on-miss primitive to make the row, then forces
    the status the test wants and commits so it is durable state the
    primitive under test reads, not staged-but-uncommitted.
    """
    topic = get_or_create_topic(db, path)
    topic.status = status
    db.commit()


def test_marks_existing_topic_for_revision(db: DbSession) -> None:
    """An existing topic's status flips to needs_revision and is flush-visible.

    The function flushes but does not commit. A fresh query in the
    same session sees the new status because the flush made it
    visible.
    """
    path = "Python > Async > Coroutines"
    _seed_topic(db, path, TopicStatus.LEARNED)

    returned = mark_topic_for_revision(db, path)

    assert returned.path == path
    assert returned.status is TopicStatus.NEEDS_REVISION
    refetched = db.query(Topic).filter(Topic.path == path).one()
    assert refetched.status is TopicStatus.NEEDS_REVISION


def test_mark_for_revision_flushes_but_does_not_commit(db: DbSession) -> None:
    """The status change is staged, not committed: a rollback reverts it.

    This is the agent-path guarantee. The orchestrator owns the
    commit, so the core must only flush. If it wrongly committed, the
    rollback below would not restore the prior status.
    """
    path = "Python > Async > EventLoop"
    _seed_topic(db, path, TopicStatus.LEARNED)

    mark_topic_for_revision(db, path)
    assert db.query(Topic).filter(Topic.path == path).one().status is TopicStatus.NEEDS_REVISION

    db.rollback()

    reverted = db.query(Topic).filter(Topic.path == path).one().status
    assert reverted is TopicStatus.LEARNED


def test_already_needs_revision_is_idempotent(db: DbSession) -> None:
    """Marking a topic that is already needs_revision leaves it unchanged.

    No error, no duplicate, status stays needs_revision. The core is
    a plain assignment so this is a property of the assignment, not
    special-cased, but the test pins it because the planner may
    propose a topic the user already flagged.
    """
    path = "Python > Async > Tasks"
    _seed_topic(db, path, TopicStatus.NEEDS_REVISION)

    returned = mark_topic_for_revision(db, path)

    assert returned.status is TopicStatus.NEEDS_REVISION


def test_absent_path_raises_not_found(db: DbSession) -> None:
    """A path with no matching topic raises TopicNotFoundError.

    Strict by design: marking a nonexistent topic for revision is
    meaningless, so the core refuses rather than creating one. The
    raised error carries the offending path.
    """
    with pytest.raises(TopicNotFoundError) as exc_info:
        mark_topic_for_revision(db, "Nonexistent > Topic > Path")

    assert exc_info.value.path == "Nonexistent > Topic > Path"


def test_absent_path_creates_nothing(db: DbSession) -> None:
    """The strict primitive does not create a row on miss.

    Distinct from get_or_create_topic. After the raise, no topic at
    that path exists, confirming the lookup did not fall through to a
    create.
    """
    path = "Nonexistent > Topic > Path"
    with pytest.raises(TopicNotFoundError):
        mark_topic_for_revision(db, path)

    assert db.query(Topic).filter(Topic.path == path).one_or_none() is None
