"""Tests for the diagnostic tool handlers.

Covers get_weak_topics and get_stale_topics. Mirrors the helper
patterns in test_knowledge_service.py: file-local helpers for
seeding rows, no commits, db.flush() after each insert.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from app.models import (
    Difficulty,
    GradingVerdict,
    LearnedItem,
    LearnedItemStatus,
    LearningMode,
    Session,
    SessionState,
    Topic,
    TopicStatus,
    TransportKind,
)
from app.schemas.tools import (
    GetStaleTopicsInput,
    GetWeakTopicsInput,
)
from app.services.tools.handlers import (
    get_stale_topics,
    get_weak_topics,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession


# ---------- helpers ----------


def _add_topic(
    db: DbSession,
    path: str,
    last_reviewed_at: datetime | None = None,
) -> Topic:
    topic = Topic(
        path=path,
        domain=path.split(" > ", 1)[0],
        name=path.rsplit(" > ", 1)[-1],
        status=TopicStatus.IN_PROGRESS,
        last_reviewed_at=last_reviewed_at,
    )
    db.add(topic)
    db.flush()
    return topic


def _add_session(db: DbSession) -> Session:
    session = Session(
        topic_id=None,
        mode_used=LearningMode.FLASHCARD,
        state=SessionState.IN_PROGRESS,
        transport_kind=TransportKind.DEEPSEEK,
        claude_chat_url=None,
        claude_chat_message_count=0,
        active_preferences=[],
        context_snapshot={},
    )
    db.add(session)
    db.flush()
    return session


def _add_item_with_verdict(
    db: DbSession,
    session: Session,
    topic: Topic,
    verdict: GradingVerdict | None,
    *,
    question: str = "q",
    difficulty: Difficulty | None = Difficulty.BEGINNER,
) -> LearnedItem:
    item = LearnedItem(
        session_id=session.id,
        topic_id=topic.id,
        question=question,
        answer="a",
        your_answer="a",
        mode=LearningMode.FLASHCARD,
        difficulty=difficulty,
        grading_verdict=verdict,
        status=LearnedItemStatus.LEARNED,
        last_reviewed_at=datetime.now(UTC),
    )
    db.add(item)
    db.flush()
    return item


# ---------- get_weak_topics ----------


async def test_weak_topics_empty_when_no_learned_items(db: DbSession) -> None:
    output = await get_weak_topics(db, GetWeakTopicsInput())
    assert output.topics == []


async def test_weak_topics_skips_topics_with_only_correct_verdicts(db: DbSession) -> None:
    """A topic with all-correct verdicts has nothing to diagnose."""
    session = _add_session(db)
    topic = _add_topic(db, "Python > Basics")
    for _ in range(5):
        _add_item_with_verdict(db, session, topic, GradingVerdict.CORRECT)

    output = await get_weak_topics(db, GetWeakTopicsInput())

    assert output.topics == []


async def test_weak_topics_respects_min_attempts(db: DbSession) -> None:
    """Topics below min_attempts are filtered out even when wrong."""
    session = _add_session(db)
    topic = _add_topic(db, "Python > Basics")
    _add_item_with_verdict(db, session, topic, GradingVerdict.INCORRECT)

    output = await get_weak_topics(db, GetWeakTopicsInput(min_attempts=2))

    assert output.topics == []


async def test_weak_topics_skips_items_without_verdict(db: DbSession) -> None:
    """Pre-split learned items have grading_verdict=None and shouldn't count."""
    session = _add_session(db)
    topic = _add_topic(db, "Python > Basics")
    for _ in range(5):
        _add_item_with_verdict(db, session, topic, None)

    output = await get_weak_topics(db, GetWeakTopicsInput())

    assert output.topics == []


async def test_weak_topics_reports_counts_per_verdict(db: DbSession) -> None:
    session = _add_session(db)
    topic = _add_topic(db, "Python > Basics")
    for _ in range(3):
        _add_item_with_verdict(db, session, topic, GradingVerdict.INCORRECT)
    for _ in range(2):
        _add_item_with_verdict(db, session, topic, GradingVerdict.PARTIAL)
    _add_item_with_verdict(db, session, topic, GradingVerdict.CORRECT)

    output = await get_weak_topics(db, GetWeakTopicsInput())

    assert len(output.topics) == 1
    weak = output.topics[0]
    assert weak.topic_path == "Python > Basics"
    assert weak.incorrect_count == 3
    assert weak.partial_count == 2
    assert weak.correct_count == 1


async def test_weak_topics_orders_by_weakness_score_worst_first(db: DbSession) -> None:
    """Score formula: (incorrect + 0.5·partial) / total. Worst score first."""
    session = _add_session(db)

    # Topic A: 4 incorrect, 1 correct -> score 0.8
    topic_a = _add_topic(db, "Python > A")
    for _ in range(4):
        _add_item_with_verdict(db, session, topic_a, GradingVerdict.INCORRECT)
    _add_item_with_verdict(db, session, topic_a, GradingVerdict.CORRECT)

    # Topic B: 1 incorrect, 4 correct -> score 0.2
    topic_b = _add_topic(db, "Python > B")
    _add_item_with_verdict(db, session, topic_b, GradingVerdict.INCORRECT)
    for _ in range(4):
        _add_item_with_verdict(db, session, topic_b, GradingVerdict.CORRECT)

    output = await get_weak_topics(db, GetWeakTopicsInput())

    assert [t.topic_path for t in output.topics] == ["Python > A", "Python > B"]


async def test_weak_topics_includes_sample_questions_for_wrong_answers(db: DbSession) -> None:
    """sample_size controls how many representative wrong-answer questions return."""
    session = _add_session(db)
    topic = _add_topic(db, "Python > Basics")
    _add_item_with_verdict(
        db, session, topic, GradingVerdict.INCORRECT, question="What is a list comprehension?"
    )
    _add_item_with_verdict(
        db, session, topic, GradingVerdict.PARTIAL, question="Explain map and filter."
    )
    _add_item_with_verdict(db, session, topic, GradingVerdict.CORRECT, question="What is None?")

    output = await get_weak_topics(db, GetWeakTopicsInput(sample_size=3))

    assert len(output.topics) == 1
    samples = output.topics[0].samples
    assert len(samples) == 2  # only INCORRECT and PARTIAL contribute
    sample_questions = {s.question for s in samples}
    assert "What is a list comprehension?" in sample_questions
    assert "Explain map and filter." in sample_questions
    assert "What is None?" not in sample_questions


async def test_weak_topics_sample_size_zero_returns_no_samples(db: DbSession) -> None:
    session = _add_session(db)
    topic = _add_topic(db, "Python > Basics")
    for _ in range(3):
        _add_item_with_verdict(db, session, topic, GradingVerdict.INCORRECT)

    output = await get_weak_topics(db, GetWeakTopicsInput(sample_size=0))

    assert len(output.topics) == 1
    assert output.topics[0].samples == []


async def test_weak_topics_truncates_long_questions_at_200_chars(db: DbSession) -> None:
    """200-char cap with ellipsis on truncation."""
    session = _add_session(db)
    topic = _add_topic(db, "Python > Basics")
    long_question = "x" * 500
    for _ in range(2):
        _add_item_with_verdict(db, session, topic, GradingVerdict.INCORRECT, question=long_question)

    output = await get_weak_topics(db, GetWeakTopicsInput(sample_size=2))

    for sample in output.topics[0].samples:
        assert len(sample.question) == 200
        assert sample.question.endswith("…")


async def test_weak_topics_caps_sample_count(db: DbSession) -> None:
    """More wrong answers than sample_size: cap is respected."""
    session = _add_session(db)
    topic = _add_topic(db, "Python > Basics")
    for i in range(10):
        _add_item_with_verdict(db, session, topic, GradingVerdict.INCORRECT, question=f"q{i}")

    output = await get_weak_topics(db, GetWeakTopicsInput(sample_size=3))

    assert len(output.topics[0].samples) == 3


async def test_weak_topics_groups_across_sessions(db: DbSession) -> None:
    """Multiple sessions on the same topic aggregate into one row."""
    topic = _add_topic(db, "Python > Basics")
    s1 = _add_session(db)
    s2 = _add_session(db)
    for _ in range(2):
        _add_item_with_verdict(db, s1, topic, GradingVerdict.INCORRECT)
    for _ in range(2):
        _add_item_with_verdict(db, s2, topic, GradingVerdict.INCORRECT)

    output = await get_weak_topics(db, GetWeakTopicsInput())

    assert len(output.topics) == 1
    assert output.topics[0].incorrect_count == 4


# ---------- get_stale_topics ----------


async def test_stale_topics_empty_when_no_topics(db: DbSession) -> None:
    output = await get_stale_topics(db, GetStaleTopicsInput())
    assert output.topics == []


async def test_stale_topics_skips_topics_never_reviewed(db: DbSession) -> None:
    """Topics with last_reviewed_at IS NULL are not stale, they're un-reviewed."""
    _add_topic(db, "Python > Basics", last_reviewed_at=None)

    output = await get_stale_topics(db, GetStaleTopicsInput(days_threshold=1))

    assert output.topics == []


async def test_stale_topics_skips_recently_reviewed(db: DbSession) -> None:
    now = datetime.now(UTC)
    _add_topic(db, "Python > Basics", last_reviewed_at=now - timedelta(days=3))

    output = await get_stale_topics(db, GetStaleTopicsInput(days_threshold=14))

    assert output.topics == []


async def test_stale_topics_returns_old_topics(db: DbSession) -> None:
    now = datetime.now(UTC)
    _add_topic(db, "Python > Old", last_reviewed_at=now - timedelta(days=30))

    output = await get_stale_topics(db, GetStaleTopicsInput(days_threshold=14))

    assert len(output.topics) == 1
    stale = output.topics[0]
    assert stale.topic_path == "Python > Old"
    assert stale.days_since_review == 30


async def test_stale_topics_ordered_oldest_first(db: DbSession) -> None:
    now = datetime.now(UTC)
    _add_topic(db, "Python > Newer", last_reviewed_at=now - timedelta(days=20))
    _add_topic(db, "Python > Older", last_reviewed_at=now - timedelta(days=60))
    _add_topic(db, "Python > Middle", last_reviewed_at=now - timedelta(days=40))

    output = await get_stale_topics(db, GetStaleTopicsInput(days_threshold=14))

    assert [t.topic_path for t in output.topics] == [
        "Python > Older",
        "Python > Middle",
        "Python > Newer",
    ]


async def test_stale_topics_respects_limit(db: DbSession) -> None:
    now = datetime.now(UTC)
    for i in range(10):
        _add_topic(db, f"Python > T{i}", last_reviewed_at=now - timedelta(days=30 + i))

    output = await get_stale_topics(db, GetStaleTopicsInput(days_threshold=14, limit=3))

    assert len(output.topics) == 3
