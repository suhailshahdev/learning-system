"""Tests for knowledge_service."""

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
)
from app.services.knowledge_service import (
    DERIVATION_THRESHOLD,
    derive_assertions_for_session,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession


def _add_topic(db: DbSession, path: str) -> Topic:
    topic = Topic(
        path=path,
        domain=path.split(" > ", 1)[0],
        name=path.rsplit(" > ", 1)[-1],
        status=TopicStatus.IN_PROGRESS,
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


def _add_item(
    db: DbSession,
    session: Session,
    topic: Topic,
    difficulty: Difficulty | None = Difficulty.BEGINNER,
) -> LearnedItem:
    item = LearnedItem(
        session_id=session.id,
        topic_id=topic.id,
        question="q",
        answer="a",
        your_answer="a",
        mode=LearningMode.FLASHCARD,
        difficulty=difficulty,
        status=LearnedItemStatus.LEARNED,
        last_reviewed_at=datetime.now(UTC),
    )
    db.add(item)
    db.flush()
    return item


def test_session_with_no_items_derives_nothing(db: DbSession) -> None:
    session = _add_session(db)

    upserts = derive_assertions_for_session(db, session)

    assert upserts == []
    assert db.query(UserKnowledgeAssertion).count() == 0


def test_below_threshold_derives_nothing(db: DbSession) -> None:
    session = _add_session(db)
    topic = _add_topic(db, "Python > Basics")
    for _ in range(DERIVATION_THRESHOLD - 1):
        _add_item(db, session, topic, Difficulty.BEGINNER)

    upserts = derive_assertions_for_session(db, session)

    assert upserts == []
    assert db.query(UserKnowledgeAssertion).count() == 0


def test_at_threshold_creates_assertion(db: DbSession) -> None:
    session = _add_session(db)
    topic = _add_topic(db, "Python > Basics")
    for _ in range(DERIVATION_THRESHOLD):
        _add_item(db, session, topic, Difficulty.BEGINNER)

    upserts = derive_assertions_for_session(db, session)

    assert len(upserts) == 1
    assertion = upserts[0]
    assert assertion.topic_path == "Python > Basics"
    assert assertion.difficulty == Difficulty.BEGINNER
    assert assertion.source == AssertionSource.DERIVED_FROM_LEARNED_ITEMS


def test_higher_difficulty_wins_when_both_meet_threshold(db: DbSession) -> None:
    session = _add_session(db)
    topic = _add_topic(db, "Python > Basics")
    for _ in range(DERIVATION_THRESHOLD):
        _add_item(db, session, topic, Difficulty.BEGINNER)
    for _ in range(DERIVATION_THRESHOLD):
        _add_item(db, session, topic, Difficulty.INTERMEDIATE)

    upserts = derive_assertions_for_session(db, session)

    assert len(upserts) == 1
    assert upserts[0].difficulty == Difficulty.INTERMEDIATE


def test_null_difficulty_items_do_not_count(db: DbSession) -> None:
    session = _add_session(db)
    topic = _add_topic(db, "Python > Basics")
    for _ in range(DERIVATION_THRESHOLD):
        _add_item(db, session, topic, None)

    upserts = derive_assertions_for_session(db, session)

    assert upserts == []
    assert db.query(UserKnowledgeAssertion).count() == 0


def test_subsequent_session_upgrades_existing_assertion(db: DbSession) -> None:
    """Second approval at a higher difficulty upgrades the derived assertion."""
    topic = _add_topic(db, "Python > Basics")

    first_session = _add_session(db)
    for _ in range(DERIVATION_THRESHOLD):
        _add_item(db, first_session, topic, Difficulty.BEGINNER)
    derive_assertions_for_session(db, first_session)

    assert db.query(UserKnowledgeAssertion).count() == 1
    initial = db.query(UserKnowledgeAssertion).one()
    assert initial.difficulty == Difficulty.BEGINNER

    second_session = _add_session(db)
    for _ in range(DERIVATION_THRESHOLD):
        _add_item(db, second_session, topic, Difficulty.INTERMEDIATE)
    upserts = derive_assertions_for_session(db, second_session)

    assert len(upserts) == 1
    # Re-queries rather than refreshing the local object. A refresh and
    # compare pattern reads as statically unreachable to mypy due to
    # literal-equality narrowing.
    upgraded = db.query(UserKnowledgeAssertion).one()
    assert upgraded.difficulty == Difficulty.INTERMEDIATE
    # Still one row; the existing assertion was updated, not duplicated.
    assert db.query(UserKnowledgeAssertion).count() == 1


def test_subsequent_session_at_lower_difficulty_does_not_downgrade(db: DbSession) -> None:
    """A later session at a lower difficulty leaves the derived row alone."""
    topic = _add_topic(db, "Python > Basics")

    first_session = _add_session(db)
    for _ in range(DERIVATION_THRESHOLD):
        _add_item(db, first_session, topic, Difficulty.INTERMEDIATE)
    derive_assertions_for_session(db, first_session)

    second_session = _add_session(db)
    for _ in range(DERIVATION_THRESHOLD):
        _add_item(db, second_session, topic, Difficulty.BEGINNER)
    upserts = derive_assertions_for_session(db, second_session)

    assert upserts == []
    existing = db.query(UserKnowledgeAssertion).one()
    assert existing.difficulty == Difficulty.INTERMEDIATE


def test_self_declared_assertion_is_untouched(db: DbSession) -> None:
    """Derivation never modifies assertions from other sources."""
    topic = _add_topic(db, "Python > Basics")
    db.add(
        UserKnowledgeAssertion(
            topic_path="Python > Basics",
            difficulty=Difficulty.ADVANCED,
            source=AssertionSource.SELF_DECLARED,
        )
    )
    db.commit()

    session = _add_session(db)
    for _ in range(DERIVATION_THRESHOLD):
        _add_item(db, session, topic, Difficulty.BEGINNER)

    upserts = derive_assertions_for_session(db, session)

    # Derived assertion was created at beginner.
    assert len(upserts) == 1
    assert upserts[0].source == AssertionSource.DERIVED_FROM_LEARNED_ITEMS
    assert upserts[0].difficulty == Difficulty.BEGINNER

    # Self-declared assertion still sits at advanced, untouched.
    self_declared = (
        db.query(UserKnowledgeAssertion)
        .filter(UserKnowledgeAssertion.source == AssertionSource.SELF_DECLARED)
        .one()
    )
    assert self_declared.difficulty == Difficulty.ADVANCED


def test_cross_topic_session_derives_per_topic(db: DbSession) -> None:
    """A session that minted items across topics derives independently for each."""
    session = _add_session(db)
    py = _add_topic(db, "Python > Basics")
    api = _add_topic(db, "FastAPI > Routing")

    for _ in range(DERIVATION_THRESHOLD):
        _add_item(db, session, py, Difficulty.BEGINNER)
    for _ in range(DERIVATION_THRESHOLD):
        _add_item(db, session, api, Difficulty.INTERMEDIATE)

    upserts = derive_assertions_for_session(db, session)

    assert len(upserts) == 2
    by_path = {a.topic_path: a for a in upserts}
    assert by_path["Python > Basics"].difficulty == Difficulty.BEGINNER
    assert by_path["FastAPI > Routing"].difficulty == Difficulty.INTERMEDIATE
