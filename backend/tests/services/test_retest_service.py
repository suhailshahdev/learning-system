"""Tests for the retest service.

Each test seeds a source session with LearnedItems, calls
start_retest, and asserts on the new session and synthetic turn.
get_next_retest_turn tests exercise the answered-count walk.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from app.models import (
    Difficulty,
    GradingVerdict,
    LearnedItem,
    LearnedItemStatus,
    LearningMode,
    Session,
    SessionState,
    SessionTurn,
    Topic,
    TopicStatus,
    TransportKind,
    TurnRole,
)
from app.services.retest_service import (
    RetestServiceError,
    get_next_retest_turn,
    start_retest,
)
from app.services.session_service import OPEN_ANSWER_PLACEHOLDER

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession


def _make_topic(db: DbSession, path: str = "Python > Data Types > Integers") -> Topic:
    """Seed a topic for use in source sessions and items."""
    domain = path.split(" > ", 1)[0]
    name = path.rsplit(" > ", 1)[-1]
    topic = Topic(path=path, domain=domain, name=name, status=TopicStatus.LEARNED)
    db.add(topic)
    db.flush()
    return topic


def _make_source_session(
    db: DbSession,
    *,
    topic_id: str | None,
    state: SessionState = SessionState.COMPLETED,
) -> Session:
    """Seed a source session in the given state (default COMPLETED)."""
    session = Session(
        topic_id=topic_id,
        mode_used=LearningMode.FLASHCARD,
        state=state,
        transport_kind=TransportKind.DEEPSEEK,
        active_preferences=[],
        context_snapshot={},
    )
    db.add(session)
    db.flush()
    return session


def _make_learned_item(
    db: DbSession,
    *,
    session_id: str,
    topic_id: str,
    question: str = "What is an integer?",
    answer: str = "A whole number.",
    mode: LearningMode = LearningMode.FLASHCARD,
    difficulty: Difficulty | None = Difficulty.BEGINNER,
    verdict: GradingVerdict | None = None,
    created_at: datetime | None = None,
) -> LearnedItem:
    """Seed one learned item on the source session."""
    item = LearnedItem(
        session_id=session_id,
        topic_id=topic_id,
        question=question,
        answer=answer,
        your_answer="prior answer",
        mode=mode,
        difficulty=difficulty,
        grading_verdict=verdict,
        status=LearnedItemStatus.LEARNED,
        created_at=created_at or datetime(2026, 5, 1, tzinfo=UTC),
    )
    db.add(item)
    db.flush()
    return item


def _add_assistant_turn(
    db: DbSession,
    *,
    session_id: str,
    turn_index: int,
    topic_path: str,
    question: str,
) -> SessionTurn:
    """Seed an ASSISTANT teaching turn on a retest session."""
    parsed: dict[str, object] = {
        "kind": "turn",
        "topic_path": topic_path,
        "difficulty": "beginner",
        "prerequisites": [],
        "mode": "flashcard",
        "question": question,
        "question_code": None,
        "expected_answer": "A whole number.",
        "requirements": None,
        "followup": None,
        "tags": [],
    }
    turn = SessionTurn(
        session_id=session_id,
        turn_index=turn_index,
        role=TurnRole.ASSISTANT,
        raw_content="<retest>",
        parsed=parsed,
        mode=LearningMode.FLASHCARD,
    )
    db.add(turn)
    db.flush()
    return turn


def _add_user_turn(db: DbSession, *, session_id: str, turn_index: int) -> SessionTurn:
    """Seed a USER answer turn on a retest session."""
    turn = SessionTurn(
        session_id=session_id,
        turn_index=turn_index,
        role=TurnRole.USER,
        raw_content="user answer",
        parsed=None,
        mode=None,
    )
    db.add(turn)
    db.flush()
    return turn


async def test_start_retest_creates_session_linked_to_source(db: DbSession) -> None:
    """start_retest creates an IN_PROGRESS session with parent_session_id set."""
    topic = _make_topic(db)
    source = _make_source_session(db, topic_id=topic.id)
    _make_learned_item(db, session_id=source.id, topic_id=topic.id)
    db.commit()

    retest_session, first_turn = start_retest(
        db=db,
        source_session_id=source.id,
        transport_kind=TransportKind.DEEPSEEK,
    )

    assert retest_session.id != source.id
    assert retest_session.parent_session_id == source.id
    assert retest_session.state is SessionState.IN_PROGRESS
    assert retest_session.topic_id == topic.id
    assert retest_session.transport_kind is TransportKind.DEEPSEEK
    assert retest_session.claude_chat_url is None
    assert retest_session.claude_chat_message_count == 0
    # First turn reconstructed from the source's first item.
    assert first_turn.question == "What is an integer?"
    assert first_turn.topic_path == "Python > Data Types > Integers"


async def test_start_retest_persists_synthetic_assistant_turn(db: DbSession) -> None:
    """The first source item materializes as an ASSISTANT turn at index 0."""
    topic = _make_topic(db)
    source = _make_source_session(db, topic_id=topic.id)
    _make_learned_item(db, session_id=source.id, topic_id=topic.id)
    db.commit()

    retest_session, _ = start_retest(
        db=db,
        source_session_id=source.id,
        transport_kind=TransportKind.DEEPSEEK,
    )
    db.refresh(retest_session)

    turns = sorted(retest_session.turns, key=lambda t: t.turn_index)
    assert len(turns) == 1
    assert turns[0].turn_index == 0
    assert turns[0].role is TurnRole.ASSISTANT
    assert turns[0].parsed is not None
    assert turns[0].parsed["kind"] == "turn"
    assert turns[0].parsed["question"] == "What is an integer?"


async def test_start_retest_picks_oldest_source_item_first(db: DbSession) -> None:
    """Source items walked by created_at ascending.

    Falsifying test for the ordering rule: insert items
    out of created_at order and confirm the synthetic turn
    comes from the oldest one regardless of insertion order.
    """
    topic = _make_topic(db)
    source = _make_source_session(db, topic_id=topic.id)
    # Insert newer item first to prove the service sorts.
    _make_learned_item(
        db,
        session_id=source.id,
        topic_id=topic.id,
        question="Newer question",
        created_at=datetime(2026, 5, 3, tzinfo=UTC),
    )
    _make_learned_item(
        db,
        session_id=source.id,
        topic_id=topic.id,
        question="Oldest question",
        created_at=datetime(2026, 5, 1, tzinfo=UTC),
    )
    _make_learned_item(
        db,
        session_id=source.id,
        topic_id=topic.id,
        question="Middle question",
        created_at=datetime(2026, 5, 2, tzinfo=UTC),
    )
    db.commit()

    _, first_turn = start_retest(
        db=db,
        source_session_id=source.id,
        transport_kind=TransportKind.DEEPSEEK,
    )

    assert first_turn.question == "Oldest question"


async def test_start_retest_recovers_open_graded_expected_answer(db: DbSession) -> None:
    """OPEN_ANSWER_PLACEHOLDER round-trips back to None.

    Falsifying test: a source item with the placeholder string
    in its answer column came from a turn whose EXPECTED_ANSWER
    was OPEN. The retest must present expected_answer=None so
    downstream grading code recognizes it as open-graded.
    """
    topic = _make_topic(db)
    source = _make_source_session(db, topic_id=topic.id)
    _make_learned_item(
        db,
        session_id=source.id,
        topic_id=topic.id,
        mode=LearningMode.EXPLAIN_BACK,
        answer=OPEN_ANSWER_PLACEHOLDER,
    )
    db.commit()

    _, first_turn = start_retest(
        db=db,
        source_session_id=source.id,
        transport_kind=TransportKind.DEEPSEEK,
    )

    assert first_turn.expected_answer is None


async def test_start_retest_404_for_unknown_source(db: DbSession) -> None:
    """Unknown source session id raises not_found."""
    with pytest.raises(RetestServiceError) as exc_info:
        start_retest(
            db=db,
            source_session_id="does-not-exist",
            transport_kind=TransportKind.DEEPSEEK,
        )

    assert exc_info.value.kind == "not_found"


async def test_start_retest_409_for_in_progress_source(db: DbSession) -> None:
    """In-progress source session is not eligible for retest."""
    topic = _make_topic(db)
    source = _make_source_session(db, topic_id=topic.id, state=SessionState.IN_PROGRESS)
    _make_learned_item(db, session_id=source.id, topic_id=topic.id)
    db.commit()

    with pytest.raises(RetestServiceError) as exc_info:
        start_retest(
            db=db,
            source_session_id=source.id,
            transport_kind=TransportKind.DEEPSEEK,
        )

    assert exc_info.value.kind == "not_eligible"


async def test_start_retest_409_for_abandoned_source(db: DbSession) -> None:
    """Abandoned source session is not eligible: nothing to grade against."""
    topic = _make_topic(db)
    source = _make_source_session(db, topic_id=topic.id, state=SessionState.ABANDONED)
    _make_learned_item(db, session_id=source.id, topic_id=topic.id)
    db.commit()

    with pytest.raises(RetestServiceError) as exc_info:
        start_retest(
            db=db,
            source_session_id=source.id,
            transport_kind=TransportKind.DEEPSEEK,
        )

    assert exc_info.value.kind == "not_eligible"


async def test_start_retest_409_for_archived_source(db: DbSession) -> None:
    """Archived source session is not eligible until un-archive surface exists."""
    topic = _make_topic(db)
    source = _make_source_session(db, topic_id=topic.id, state=SessionState.ARCHIVED)
    _make_learned_item(db, session_id=source.id, topic_id=topic.id)
    db.commit()

    with pytest.raises(RetestServiceError) as exc_info:
        start_retest(
            db=db,
            source_session_id=source.id,
            transport_kind=TransportKind.DEEPSEEK,
        )

    assert exc_info.value.kind == "not_eligible"


async def test_start_retest_409_for_empty_source(db: DbSession) -> None:
    """A completed source with no learned items has nothing to retest."""
    topic = _make_topic(db)
    source = _make_source_session(db, topic_id=topic.id)
    db.commit()  # COMPLETED state, zero learned items

    with pytest.raises(RetestServiceError) as exc_info:
        start_retest(
            db=db,
            source_session_id=source.id,
            transport_kind=TransportKind.DEEPSEEK,
        )

    assert exc_info.value.kind == "empty_source"


async def test_get_next_retest_turn_returns_second_item_after_first_answered(
    db: DbSession,
) -> None:
    """After one question is answered, get_next returns the source's second item."""
    topic = _make_topic(db)
    source = _make_source_session(db, topic_id=topic.id)
    _make_learned_item(
        db,
        session_id=source.id,
        topic_id=topic.id,
        question="First question",
        created_at=datetime(2026, 5, 1, tzinfo=UTC),
    )
    _make_learned_item(
        db,
        session_id=source.id,
        topic_id=topic.id,
        question="Second question",
        created_at=datetime(2026, 5, 2, tzinfo=UTC),
    )
    db.commit()

    retest_session, _ = start_retest(
        db=db,
        source_session_id=source.id,
        transport_kind=TransportKind.DEEPSEEK,
    )
    # Simulate the user answering the first question by appending
    # a USER turn at index 1.
    _add_user_turn(db, session_id=retest_session.id, turn_index=1)
    db.commit()

    next_turn = get_next_retest_turn(db, retest_session.id)

    assert next_turn is not None
    assert next_turn.question == "Second question"


async def test_get_next_retest_turn_returns_none_when_all_answered(
    db: DbSession,
) -> None:
    """Returns None when the user has answered every source item."""
    topic = _make_topic(db)
    source = _make_source_session(db, topic_id=topic.id)
    _make_learned_item(
        db,
        session_id=source.id,
        topic_id=topic.id,
        question="Only question",
        created_at=datetime(2026, 5, 1, tzinfo=UTC),
    )
    db.commit()

    retest_session, _ = start_retest(
        db=db,
        source_session_id=source.id,
        transport_kind=TransportKind.DEEPSEEK,
    )
    _add_user_turn(db, session_id=retest_session.id, turn_index=1)
    db.commit()

    assert get_next_retest_turn(db, retest_session.id) is None


async def test_get_next_retest_turn_skips_unanswered_assistant_turns(
    db: DbSession,
) -> None:
    """Answered count tracks USER turns following ASSISTANT teaching turns.

    A trailing ASSISTANT turn without a USER follow-up does not
    count as answered. Without this rule, the second teaching turn
    would be skipped and the third question would surface
    incorrectly.
    """
    topic = _make_topic(db)
    source = _make_source_session(db, topic_id=topic.id)
    for i in range(3):
        _make_learned_item(
            db,
            session_id=source.id,
            topic_id=topic.id,
            question=f"Question {i}",
            created_at=datetime(2026, 5, 1, tzinfo=UTC) + timedelta(minutes=i),
        )
    db.commit()

    retest_session, _ = start_retest(
        db=db,
        source_session_id=source.id,
        transport_kind=TransportKind.DEEPSEEK,
    )
    # Answer first question.
    _add_user_turn(db, session_id=retest_session.id, turn_index=1)
    # Show second question (synthetic) but don't answer it yet.
    _add_assistant_turn(
        db,
        session_id=retest_session.id,
        turn_index=2,
        topic_path=topic.path,
        question="Question 1",
    )
    db.commit()

    next_turn = get_next_retest_turn(db, retest_session.id)

    # Second question is shown but unanswered, so the next item
    # to surface is still "Question 1" — the second source item.
    assert next_turn is not None
    assert next_turn.question == "Question 1"


async def test_get_next_retest_turn_404_for_unknown_session(db: DbSession) -> None:
    """Unknown retest session id raises not_found."""
    with pytest.raises(RetestServiceError) as exc_info:
        get_next_retest_turn(db, "does-not-exist")

    assert exc_info.value.kind == "not_found"


async def test_get_next_retest_turn_409_for_non_retest_session(db: DbSession) -> None:
    """A normal (non-retest) session raises not_eligible."""
    topic = _make_topic(db)
    session = _make_source_session(db, topic_id=topic.id)
    db.commit()

    with pytest.raises(RetestServiceError) as exc_info:
        get_next_retest_turn(db, session.id)

    assert exc_info.value.kind == "not_eligible"
