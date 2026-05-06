"""Tests for the home dashboard service.

Each test seeds rows directly to control field values precisely
(timestamps, status, ordering). No helper factories yet — extract
when setup repetition becomes painful.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from app.models import (
    AssertionSource,
    Difficulty,
    LearnedItem,
    LearnedItemStatus,
    LearningMode,
    Session,
    SessionState,
    Topic,
    TopicStatus,
    TransportKind,
    UserKnowledgeAssertion,
    UserProfile,
)
from app.models.user_profile import SINGLETON_ID
from app.services.home_service import build_home_response

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession


def _make_topic(
    db: DbSession,
    *,
    path: str,
    status: TopicStatus = TopicStatus.IN_PROGRESS,
) -> Topic:
    """Seed a topic with domain derived from path's first segment."""
    domain = path.split(" > ", 1)[0]
    name = path.rsplit(" > ", 1)[-1]
    topic = Topic(path=path, domain=domain, name=name, status=status)
    db.add(topic)
    db.flush()
    return topic


def _make_session(
    db: DbSession,
    *,
    topic_id: str | None = None,
    state: SessionState = SessionState.IN_PROGRESS,
    created_at: datetime,
    transport_kind: TransportKind = TransportKind.DEEPSEEK,
    mode_used: LearningMode = LearningMode.FLASHCARD,
) -> Session:
    """Seed a session with explicit created_at for deterministic ordering."""
    session = Session(
        topic_id=topic_id,
        mode_used=mode_used,
        state=state,
        transport_kind=transport_kind,
        active_preferences=[],
        context_snapshot={},
        created_at=created_at,
    )
    db.add(session)
    db.flush()
    return session


def _make_learned_item(
    db: DbSession,
    *,
    session_id: str,
    topic_id: str,
    question: str,
    last_reviewed_at: datetime | None,
    difficulty: Difficulty = Difficulty.BEGINNER,
    mode: LearningMode = LearningMode.FLASHCARD,
) -> LearnedItem:
    """Seed a learned item with explicit last_reviewed_at."""
    item = LearnedItem(
        session_id=session_id,
        topic_id=topic_id,
        question=question,
        answer="canonical answer",
        your_answer="user answer",
        mode=mode,
        difficulty=difficulty,
        status=LearnedItemStatus.LEARNED,
        last_reviewed_at=last_reviewed_at,
    )
    db.add(item)
    db.flush()
    return item


async def test_empty_db_is_blank_slate(db: DbSession) -> None:
    """Empty database: blank-slate true, all collections empty, continue_last None."""
    response = await build_home_response(db)

    assert response.is_blank_slate is True
    assert response.continue_last is None
    assert response.due_for_review == []
    assert response.focus_by_domain == []
    assert response.recent_sessions == []
    assert response.knowledge_summary == []


async def test_profile_with_default_stack_disqualifies_blank_slate(db: DbSession) -> None:
    """Any one configured profile field flips blank-slate to false."""
    db.add(UserProfile(id=SINGLETON_ID, name="ken", default_stack="Python"))
    db.commit()

    response = await build_home_response(db)

    assert response.is_blank_slate is False


async def test_learned_items_disqualify_blank_slate(db: DbSession) -> None:
    """Any learned item flips blank-slate to false even with empty profile."""
    topic = _make_topic(db, path="Python > Data Types > Integers")
    session = _make_session(
        db,
        topic_id=topic.id,
        created_at=datetime(2026, 4, 1, tzinfo=UTC),
    )
    _make_learned_item(
        db,
        session_id=session.id,
        topic_id=topic.id,
        question="q1",
        last_reviewed_at=datetime(2026, 4, 1, tzinfo=UTC),
    )
    db.commit()

    response = await build_home_response(db)

    assert response.is_blank_slate is False


async def test_continue_last_picks_most_recent_in_progress(db: DbSession) -> None:
    """The most recent IN_PROGRESS session wins, regardless of other sessions."""
    topic = _make_topic(db, path="Python > Data Types > Integers")
    _make_session(
        db,
        topic_id=topic.id,
        state=SessionState.COMPLETED,
        created_at=datetime(2026, 5, 1, tzinfo=UTC),
    )
    older_in_progress = _make_session(
        db,
        topic_id=topic.id,
        state=SessionState.IN_PROGRESS,
        created_at=datetime(2026, 4, 1, tzinfo=UTC),
    )
    newer_in_progress = _make_session(
        db,
        topic_id=topic.id,
        state=SessionState.IN_PROGRESS,
        created_at=datetime(2026, 4, 15, tzinfo=UTC),
    )
    db.commit()

    response = await build_home_response(db)

    assert response.continue_last is not None
    assert response.continue_last.id == newer_in_progress.id
    # Sanity check: not the older in-progress one and not the newer completed one
    assert response.continue_last.id != older_in_progress.id


async def test_continue_last_is_none_when_no_in_progress(db: DbSession) -> None:
    """Only completed/abandoned sessions: continue_last is None."""
    topic = _make_topic(db, path="Python > Data Types > Integers")
    _make_session(
        db,
        topic_id=topic.id,
        state=SessionState.COMPLETED,
        created_at=datetime(2026, 5, 1, tzinfo=UTC),
    )
    _make_session(
        db,
        topic_id=topic.id,
        state=SessionState.ABANDONED,
        created_at=datetime(2026, 4, 1, tzinfo=UTC),
    )
    db.commit()

    response = await build_home_response(db)

    assert response.continue_last is None


async def test_due_for_review_sorts_by_last_reviewed_nulls_last(db: DbSession) -> None:
    """Items sort oldest review first. Items that have never been reviewed sort last."""
    topic = _make_topic(db, path="Python > Data Types > Integers")
    session = _make_session(
        db,
        topic_id=topic.id,
        created_at=datetime(2026, 4, 1, tzinfo=UTC),
    )
    # Three items: oldest review, recent review, never reviewed.
    oldest = _make_learned_item(
        db,
        session_id=session.id,
        topic_id=topic.id,
        question="oldest",
        last_reviewed_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    recent = _make_learned_item(
        db,
        session_id=session.id,
        topic_id=topic.id,
        question="recent",
        last_reviewed_at=datetime(2026, 4, 1, tzinfo=UTC),
    )
    never = _make_learned_item(
        db,
        session_id=session.id,
        topic_id=topic.id,
        question="never",
        last_reviewed_at=None,
    )
    db.commit()

    response = await build_home_response(db)

    assert len(response.due_for_review) == 3
    # Oldest review first, then recent, then never-reviewed at the end.
    assert response.due_for_review[0].id == oldest.id
    assert response.due_for_review[1].id == recent.id
    assert response.due_for_review[2].id == never.id
    # Each summary carries the topic path
    assert response.due_for_review[0].topic_path == "Python > Data Types > Integers"


async def test_focus_by_domain_groups_in_progress_topics(db: DbSession) -> None:
    """In-progress topics group by domain; non-in-progress are excluded."""
    _make_topic(db, path="Python > Data Types > Integers", status=TopicStatus.IN_PROGRESS)
    _make_topic(db, path="Python > Functions > Closures", status=TopicStatus.IN_PROGRESS)
    _make_topic(db, path="FastAPI > Routing > Path Parameters", status=TopicStatus.IN_PROGRESS)
    _make_topic(db, path="Python > Old Stuff", status=TopicStatus.LEARNED)  # excluded
    _make_topic(db, path="React > Hooks > useState", status=TopicStatus.NOT_STARTED)  # excluded
    db.commit()

    response = await build_home_response(db)

    # Two domains have in-progress topics. React/Python-LEARNED are excluded.
    assert len(response.focus_by_domain) == 2
    by_domain = {f.domain: f for f in response.focus_by_domain}
    assert "Python" in by_domain
    assert "FastAPI" in by_domain

    # Python has two in-progress topics and FastAPI has one.
    assert len(by_domain["Python"].in_progress_topics) == 2
    assert len(by_domain["FastAPI"].in_progress_topics) == 1

    # Topics within domain are sorted by path
    python_paths = [t.path for t in by_domain["Python"].in_progress_topics]
    assert python_paths == sorted(python_paths)


async def test_recent_sessions_limited_and_sorted(db: DbSession) -> None:
    """Returns up to 5 sessions, most recent first."""
    topic = _make_topic(db, path="Python > Data Types > Integers")
    # Create 7 sessions across different dates
    for day in range(1, 8):
        _make_session(
            db,
            topic_id=topic.id,
            created_at=datetime(2026, 5, day, tzinfo=UTC),
        )
    db.commit()

    response = await build_home_response(db)

    assert len(response.recent_sessions) == 5
    # Most recent first: day 7, 6, 5, 4, 3
    timestamps = [s.created_at for s in response.recent_sessions]
    assert timestamps == sorted(timestamps, reverse=True)
    # SQLite drops timezone info on DateTime columns so this compares
    # wall-clock values rather than full datetimes. The two oldest
    # seeds should not appear in the result.
    assert response.recent_sessions[0].created_at.replace(tzinfo=None) == datetime(2026, 5, 7)
    assert response.recent_sessions[-1].created_at.replace(tzinfo=None) == datetime(2026, 5, 3)


async def test_knowledge_summary_aggregates_by_domain_and_difficulty(db: DbSession) -> None:
    """Counts distinct topic_paths per (domain, difficulty), sorted deterministically."""
    # The same topic asserted from two different sources should count as one.
    db.add(
        UserKnowledgeAssertion(
            topic_path="Python > Basics",
            difficulty=Difficulty.BEGINNER,
            source=AssertionSource.SELF_DECLARED,
        )
    )
    db.add(
        UserKnowledgeAssertion(
            topic_path="Python > Basics",
            difficulty=Difficulty.BEGINNER,
            source=AssertionSource.DERIVED_FROM_LEARNED_ITEMS,
        )
    )
    # Distinct Python topics at same difficulty
    db.add(
        UserKnowledgeAssertion(
            topic_path="Python > Functions",
            difficulty=Difficulty.BEGINNER,
            source=AssertionSource.SELF_DECLARED,
        )
    )
    # Same domain, different difficulty
    db.add(
        UserKnowledgeAssertion(
            topic_path="Python > Async",
            difficulty=Difficulty.INTERMEDIATE,
            source=AssertionSource.SELF_DECLARED,
        )
    )
    # Different domain
    db.add(
        UserKnowledgeAssertion(
            topic_path="FastAPI > Routing",
            difficulty=Difficulty.BEGINNER,
            source=AssertionSource.RESUME,
        )
    )
    db.commit()

    response = await build_home_response(db)

    # Three rows expected: (FastAPI, beginner, 1), (Python, beginner, 2), (Python, intermediate, 1)
    assert len(response.knowledge_summary) == 3

    # Check sort order: domain asc, difficulty in enum order (beginner, intermediate, advanced)
    assert response.knowledge_summary[0].domain == "FastAPI"
    assert response.knowledge_summary[0].difficulty == Difficulty.BEGINNER
    assert response.knowledge_summary[0].count == 1

    assert response.knowledge_summary[1].domain == "Python"
    assert response.knowledge_summary[1].difficulty == Difficulty.BEGINNER
    # Two distinct topics: Python > Basics and Python > Functions
    # (Python > Basics asserted twice but counts once)
    assert response.knowledge_summary[1].count == 2

    assert response.knowledge_summary[2].domain == "Python"
    assert response.knowledge_summary[2].difficulty == Difficulty.INTERMEDIATE
    assert response.knowledge_summary[2].count == 1


async def test_full_happy_path_composes_all_six_pieces(db: DbSession) -> None:
    """End-to-end: profile + topics + sessions + items + assertions all populated."""
    db.add(UserProfile(id=SINGLETON_ID, name="ken", default_stack="Python"))

    python_topic = _make_topic(db, path="Python > Data Types > Integers")
    _make_topic(db, path="FastAPI > Routing > Path Parameters")

    in_progress_session = _make_session(
        db,
        topic_id=python_topic.id,
        state=SessionState.IN_PROGRESS,
        created_at=datetime(2026, 5, 1, tzinfo=UTC),
    )
    completed_session = _make_session(
        db,
        topic_id=python_topic.id,
        state=SessionState.COMPLETED,
        created_at=datetime(2026, 4, 15, tzinfo=UTC),
    )

    _make_learned_item(
        db,
        session_id=completed_session.id,
        topic_id=python_topic.id,
        question="What is 7 // 2?",
        last_reviewed_at=datetime(2026, 4, 15, tzinfo=UTC),
    )

    db.add(
        UserKnowledgeAssertion(
            topic_path="Python > Basics",
            difficulty=Difficulty.INTERMEDIATE,
            source=AssertionSource.SELF_DECLARED,
        )
    )
    db.commit()

    response = await build_home_response(db)

    # Profile is set, so not blank-slate
    assert response.is_blank_slate is False
    # Continue-last picks the in-progress session
    assert response.continue_last is not None
    assert response.continue_last.id == in_progress_session.id
    # One due-for-review item
    assert len(response.due_for_review) == 1
    assert response.due_for_review[0].topic_path == "Python > Data Types > Integers"
    # Both seeded domains appear in focus_by_domain
    domains = {f.domain for f in response.focus_by_domain}
    assert domains == {"Python", "FastAPI"}
    # Two recent sessions (in-progress + completed)
    assert len(response.recent_sessions) == 2
    # One knowledge assertion
    assert len(response.knowledge_summary) == 1
    assert response.knowledge_summary[0].domain == "Python"
    assert response.knowledge_summary[0].difficulty == Difficulty.INTERMEDIATE
