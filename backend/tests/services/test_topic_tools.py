"""Tests for the topic-management tool handlers.

Covers create_or_update_topic with focus on ancestor auto-creation.
Mirrors the helper patterns in test_diagnostic_tools.py: file-local
helpers for seeding rows, no commits, db.flush() after each insert.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.models import Difficulty, Topic, TopicStatus
from app.schemas.common import Prerequisite
from app.schemas.tools import CreateOrUpdateTopicInput
from app.services.tools.handlers import create_or_update_topic

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession


# ---------- helpers ----------


def _add_topic(
    db: DbSession,
    path: str,
    parent_id: str | None = None,
) -> Topic:
    topic = Topic(
        path=path,
        domain=path.split(" > ", 1)[0],
        name=path.rsplit(" > ", 1)[-1],
        status=TopicStatus.IN_PROGRESS,
        parent_id=parent_id,
    )
    db.add(topic)
    db.flush()
    return topic


def _get_topic(db: DbSession, path: str) -> Topic | None:
    return db.query(Topic).filter(Topic.path == path).one_or_none()


# ---------- parent_path: missing ancestor cases ----------


async def test_parent_path_missing_creates_parent_and_links(db: DbSession) -> None:
    """parent_path references a missing topic: handler creates it and wires parent_id."""
    output = await create_or_update_topic(
        db,
        CreateOrUpdateTopicInput(
            path="Python > Async > Asyncio",
            parent_path="Python > Async",
        ),
    )

    parent = _get_topic(db, "Python > Async")
    assert parent is not None
    assert output.topic.path == "Python > Async > Asyncio"

    leaf = _get_topic(db, "Python > Async > Asyncio")
    assert leaf is not None
    assert leaf.parent_id == parent.id


async def test_parent_path_chain_creates_full_chain(db: DbSession) -> None:
    """parent_path references multiple missing ancestors: handler creates the whole chain."""
    await create_or_update_topic(
        db,
        CreateOrUpdateTopicInput(
            path="Python > Async > Asyncio > EventLoop",
            parent_path="Python > Async > Asyncio",
        ),
    )

    a = _get_topic(db, "Python")
    b = _get_topic(db, "Python > Async")
    c = _get_topic(db, "Python > Async > Asyncio")
    leaf = _get_topic(db, "Python > Async > Asyncio > EventLoop")

    assert a is not None
    assert b is not None
    assert c is not None
    assert leaf is not None

    assert a.parent_id is None
    assert b.parent_id == a.id
    assert c.parent_id == b.id
    assert leaf.parent_id == c.id


async def test_parent_path_existing_topic_is_not_recreated(db: DbSession) -> None:
    """parent_path references an existing topic: same id, no duplicate row."""
    existing = _add_topic(db, "Python > Async")
    existing_id = existing.id

    await create_or_update_topic(
        db,
        CreateOrUpdateTopicInput(
            path="Python > Async > Asyncio",
            parent_path="Python > Async",
        ),
    )

    rows = db.query(Topic).filter(Topic.path == "Python > Async").all()
    assert len(rows) == 1
    assert rows[0].id == existing_id

    leaf = _get_topic(db, "Python > Async > Asyncio")
    assert leaf is not None
    assert leaf.parent_id == existing_id


async def test_parent_path_partial_chain_fills_only_missing(db: DbSession) -> None:
    """Some ancestors exist, others missing: handler creates only the missing."""
    a = _add_topic(db, "Python")
    a_id = a.id

    await create_or_update_topic(
        db,
        CreateOrUpdateTopicInput(
            path="Python > Async > Asyncio",
            parent_path="Python > Async",
        ),
    )

    a_after = _get_topic(db, "Python")
    b = _get_topic(db, "Python > Async")
    leaf = _get_topic(db, "Python > Async > Asyncio")

    assert a_after is not None
    assert a_after.id == a_id  # unchanged
    assert b is not None
    assert b.parent_id == a_id
    assert leaf is not None
    assert leaf.parent_id == b.id


# ---------- leaf path: implied ancestors ----------


async def test_leaf_path_with_missing_ancestors_creates_them(db: DbSession) -> None:
    """No parent_path supplied: handler still creates ancestors implied by leaf path."""
    output = await create_or_update_topic(
        db,
        CreateOrUpdateTopicInput(path="Python > Async > Asyncio"),
    )

    a = _get_topic(db, "Python")
    b = _get_topic(db, "Python > Async")
    leaf = _get_topic(db, "Python > Async > Asyncio")

    assert a is not None
    assert b is not None
    assert leaf is not None
    assert b.parent_id == a.id
    assert leaf.parent_id == b.id
    assert output.created is True


# ---------- metadata isolation ----------


async def test_auto_created_ancestors_have_correct_domain(db: DbSession) -> None:
    """Domain denormalizes from first path segment for every auto-created row."""
    await create_or_update_topic(
        db,
        CreateOrUpdateTopicInput(path="Python > Async > Asyncio"),
    )

    for path in ("Python", "Python > Async", "Python > Async > Asyncio"):
        topic = _get_topic(db, path)
        assert topic is not None
        assert topic.domain == "Python"


async def test_auto_created_ancestors_have_minimal_metadata(db: DbSession) -> None:
    """Ancestors inherit only schema defaults, not the leaf's metadata."""
    await create_or_update_topic(
        db,
        CreateOrUpdateTopicInput(
            path="Python > Async > Asyncio",
            difficulty=Difficulty.ADVANCED,
            prerequisites=[
                Prerequisite(topic_path="Python > Basics", min_difficulty=Difficulty.BEGINNER),
            ],
        ),
    )

    for path in ("Python", "Python > Async"):
        topic = _get_topic(db, path)
        assert topic is not None
        assert topic.difficulty is None
        assert topic.prerequisites == []
        assert topic.status == TopicStatus.IN_PROGRESS

    leaf = _get_topic(db, "Python > Async > Asyncio")
    assert leaf is not None
    assert leaf.difficulty == Difficulty.ADVANCED
    assert len(leaf.prerequisites) == 1
