"""Tests for the sessions browse service.

Each test seeds sessions and exercises the filter, sort, and
limit logic. Helpers are private to this file per the resume
and transcript test pattern.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from app.models import (
    LearnedItem,
    LearnedItemStatus,
    LearningMode,
    Session,
    SessionState,
    Topic,
    TopicStatus,
    TransportKind,
)
from app.services.browse_service import BROWSE_LIMIT, list_sessions

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession


def _make_topic(db: DbSession, path: str = "Python > Data Types > Integers") -> Topic:
    """Seed a topic for use as session.topic_id."""
    domain = path.split(" > ", 1)[0]
    name = path.rsplit(" > ", 1)[-1]
    topic = Topic(path=path, domain=domain, name=name, status=TopicStatus.LEARNED)
    db.add(topic)
    db.flush()
    return topic


def _make_session(
    db: DbSession,
    *,
    topic_id: str | None = None,
    state: SessionState = SessionState.COMPLETED,
    created_at: datetime | None = None,
) -> Session:
    """Seed a session in the given state and creation time."""
    session = Session(
        topic_id=topic_id,
        mode_used=LearningMode.FLASHCARD,
        state=state,
        transport_kind=TransportKind.DEEPSEEK,
        active_preferences=[],
        context_snapshot={},
        created_at=created_at or datetime(2026, 5, 1, tzinfo=UTC),
    )
    db.add(session)
    db.flush()
    return session


def _make_learned_item(db: DbSession, *, session_id: str, topic_id: str) -> LearnedItem:
    """Seed one learned item attached to the given session and topic."""
    item = LearnedItem(
        session_id=session_id,
        topic_id=topic_id,
        question="Q?",
        answer="A.",
        your_answer="A.",
        mode=LearningMode.FLASHCARD,
        difficulty=None,
        status=LearnedItemStatus.LEARNED,
    )
    db.add(item)
    db.flush()
    return item


async def test_browse_returns_all_sessions_sorted_by_created_at_desc(db: DbSession) -> None:
    """Browse returns sessions in created_at descending order."""
    topic = _make_topic(db)
    older = _make_session(db, topic_id=topic.id, created_at=datetime(2026, 5, 1, tzinfo=UTC))
    newer = _make_session(db, topic_id=topic.id, created_at=datetime(2026, 5, 3, tzinfo=UTC))
    middle = _make_session(db, topic_id=topic.id, created_at=datetime(2026, 5, 2, tzinfo=UTC))
    db.commit()

    response = list_sessions(db=db)

    assert len(response.rows) == 3
    assert response.rows[0].id == newer.id
    assert response.rows[1].id == middle.id
    assert response.rows[2].id == older.id
    assert response.limit_reached is False


async def test_browse_filters_by_state(db: DbSession) -> None:
    """Passing state filters to only that state."""
    topic = _make_topic(db)
    _make_session(db, topic_id=topic.id, state=SessionState.COMPLETED)
    _make_session(db, topic_id=topic.id, state=SessionState.COMPLETED)
    _make_session(db, topic_id=topic.id, state=SessionState.ABANDONED)
    _make_session(db, topic_id=topic.id, state=SessionState.IN_PROGRESS)
    db.commit()

    response = list_sessions(db=db, state=SessionState.COMPLETED)

    assert len(response.rows) == 2
    assert all(row.state == SessionState.COMPLETED for row in response.rows)


async def test_browse_includes_topic_path_via_join(db: DbSession) -> None:
    """Browse rows carry topic_path joined from the Topic table."""
    topic = _make_topic(db, path="Python > Functions > Closures")
    session = _make_session(db, topic_id=topic.id)
    db.commit()

    response = list_sessions(db=db)

    assert len(response.rows) == 1
    assert response.rows[0].id == session.id
    assert response.rows[0].topic_path == "Python > Functions > Closures"


async def test_browse_handles_topic_less_sessions(db: DbSession) -> None:
    """Sessions with topic_id=None have topic_path=None in browse rows."""
    session = _make_session(db, topic_id=None)
    db.commit()

    response = list_sessions(db=db)

    assert len(response.rows) == 1
    assert response.rows[0].id == session.id
    assert response.rows[0].topic_path is None


async def test_browse_carries_learned_item_count(db: DbSession) -> None:
    """Each row's learned_item_count matches that session's LearnedItem rows."""
    topic = _make_topic(db)
    busy = _make_session(db, topic_id=topic.id)
    empty = _make_session(
        db,
        topic_id=topic.id,
        created_at=datetime(2026, 4, 1, tzinfo=UTC),
    )
    for _ in range(3):
        _make_learned_item(db, session_id=busy.id, topic_id=topic.id)
    db.commit()

    response = list_sessions(db=db)

    assert len(response.rows) == 2
    by_id = {row.id: row for row in response.rows}
    assert by_id[busy.id].learned_item_count == 3
    assert by_id[empty.id].learned_item_count == 0


async def test_browse_returns_empty_when_no_sessions(db: DbSession) -> None:
    """No sessions seeded yields empty rows and limit_reached=False."""
    response = list_sessions(db=db)

    assert response.rows == []
    assert response.limit_reached is False


async def test_browse_caps_at_browse_limit_and_signals_reached(db: DbSession) -> None:
    """Seeding BROWSE_LIMIT + 1 sessions caps response and sets limit_reached.

    Falsifying test for the limit+1 trick. With exactly BROWSE_LIMIT
    rows, limit_reached must be False. With BROWSE_LIMIT + 1 rows,
    limit_reached must be True and rows must be capped to BROWSE_LIMIT.
    """
    topic = _make_topic(db)
    base = datetime(2026, 5, 1, tzinfo=UTC)
    for i in range(BROWSE_LIMIT + 1):
        _make_session(
            db,
            topic_id=topic.id,
            created_at=base + timedelta(minutes=i),
        )
    db.commit()

    response = list_sessions(db=db)

    assert len(response.rows) == BROWSE_LIMIT
    assert response.limit_reached is True


async def test_browse_at_exactly_limit_does_not_signal_reached(db: DbSession) -> None:
    """Exactly BROWSE_LIMIT rows leaves limit_reached=False.

    Confirms the boundary: limit_reached is True only when there
    is strictly more data past the cap, not at the cap itself.
    """
    topic = _make_topic(db)
    base = datetime(2026, 5, 1, tzinfo=UTC)
    for i in range(BROWSE_LIMIT):
        _make_session(
            db,
            topic_id=topic.id,
            created_at=base + timedelta(minutes=i),
        )
    db.commit()

    response = list_sessions(db=db)

    assert len(response.rows) == BROWSE_LIMIT
    assert response.limit_reached is False
