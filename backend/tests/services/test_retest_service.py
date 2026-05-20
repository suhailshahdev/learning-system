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
from app.schemas.parsed_response import (
    ParsedGrading,
    ParsedSessionEnd,
    ParsedTurn,
)
from app.schemas.tools import GetWeakTopicsCall, GetWeakTopicsInput
from app.services.retest_service import (
    RetestServiceError,
    answer_retest_question,
    get_next_retest_turn,
    grade_retest_answer,
    next_retest_question,
    start_retest,
)
from app.services.session_service import (
    OPEN_ANSWER_PLACEHOLDER,
    SessionServiceError,
    request_next_question,
    send_user_answer,
)
from app.transport.base import TransportError, TransportResponse

from tests.services.fakes import FakeTransport

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


# ---------- grade_retest_answer ----------


GRADING_RESPONSE_CORRECT = """\
---GRADING---
correct
---GRADING_EXPLANATION---
Right. Integer division floors the result toward negative infinity.
---GRADING_EXPLANATION_CODE---
NONE
---END---
"""

GRADING_RESPONSE_INCORRECT = """\
---GRADING---
incorrect
---GRADING_EXPLANATION---
Not quite. 7 // 2 evaluates to 3, not 3.5. The // operator
truncates the result.
---GRADING_EXPLANATION_CODE---
NONE
---END---
"""

GRADING_RESPONSE_OPEN = """\
---GRADING---
open_graded
---GRADING_EXPLANATION---
Solid explanation. You covered the lookup mechanism and the
fallback to __dict__. One thing to add next time: descriptors
fit into this picture before __dict__.
---GRADING_EXPLANATION_CODE---
NONE
---END---
"""

# A teaching-turn response shape, used to assert the service
# rejects non-grading responses.
TEACHING_TURN_RESPONSE = """\
---TOPIC---
Python > Data Types > Integers
---DIFFICULTY---
beginner
---PREREQUISITES---
NONE
---MODE---
flashcard
---QUESTION---
What is 7 // 2?
---QUESTION_CODE---
NONE
---EXPECTED_ANSWER---
3
---REQUIREMENTS---
NONE
---FOLLOWUP---
NONE
---TAGS---
---END---
"""


async def test_grade_retest_answer_returns_parsed_grading_for_correct() -> None:
    """LLM returns a CORRECT grading response."""
    transport = FakeTransport([GRADING_RESPONSE_CORRECT])

    result = await grade_retest_answer(
        transport=transport,
        question="What is 7 // 2?",
        expected_answer="3",
        user_answer="3",
    )

    assert isinstance(result, ParsedGrading)
    assert result.verdict == GradingVerdict.CORRECT


async def test_grade_retest_answer_returns_parsed_grading_for_incorrect() -> None:
    """LLM returns an INCORRECT grading response."""
    transport = FakeTransport([GRADING_RESPONSE_INCORRECT])

    result = await grade_retest_answer(
        transport=transport,
        question="What is 7 // 2?",
        expected_answer="3",
        user_answer="3.5",
    )

    assert isinstance(result, ParsedGrading)
    assert result.verdict == GradingVerdict.INCORRECT


async def test_grade_retest_answer_handles_open_graded_mode() -> None:
    """expected_answer=None passes through as open-graded. LLM returns OPEN_GRADED.

    Falsifying test for the open-graded path. If the prompt
    builder didn't recognize None, the LLM would receive a
    literal None and grade nonsensically.
    """
    transport = FakeTransport([GRADING_RESPONSE_OPEN])

    result = await grade_retest_answer(
        transport=transport,
        question="Explain how attribute lookup works in Python.",
        expected_answer=None,
        user_answer="Python first checks the instance __dict__, then walks the MRO.",
    )

    assert isinstance(result, ParsedGrading)
    assert result.verdict == GradingVerdict.OPEN_GRADED


async def test_grade_retest_answer_opens_fresh_chat_each_call() -> None:
    """Two consecutive grading calls produce two separate chats.

    Falsifying test for the fresh-chat-per-question design. If
    the service reused a chat across calls, the second call
    wouldn't appear in the transport's chats list.
    """
    transport = FakeTransport([GRADING_RESPONSE_CORRECT, GRADING_RESPONSE_INCORRECT])

    await grade_retest_answer(
        transport=transport,
        question="Q1",
        expected_answer="A",
        user_answer="A",
    )
    await grade_retest_answer(
        transport=transport,
        question="Q2",
        expected_answer="B",
        user_answer="C",
    )

    assert len(transport.chats) == 2


async def test_grade_retest_answer_passes_question_and_answer_in_prompt() -> None:
    """The chat's first message carries all three fields the LLM needs."""
    transport = FakeTransport([GRADING_RESPONSE_CORRECT])

    await grade_retest_answer(
        transport=transport,
        question="What is the GIL?",
        expected_answer="Global Interpreter Lock",
        user_answer="The thing that prevents true threading",
    )

    # FakeTransport.start_new_chat appends [intro, first_message]
    # to the chat's messages_sent. The first message is index 1.
    first_message = transport.chats[0].messages_sent[1]
    assert "What is the GIL?" in first_message
    assert "Global Interpreter Lock" in first_message
    assert "The thing that prevents true threading" in first_message


async def test_grade_retest_answer_marks_open_graded_in_prompt() -> None:
    """expected_answer=None surfaces as a NONE label in the prompt.

    The LLM needs to see that this is open-graded, not a missing
    field. The prompt builder substitutes "NONE (open-graded mode)"
    when expected_answer is None.
    """
    transport = FakeTransport([GRADING_RESPONSE_OPEN])

    await grade_retest_answer(
        transport=transport,
        question="Explain X.",
        expected_answer=None,
        user_answer="X works like Y.",
    )

    first_message = transport.chats[0].messages_sent[1]
    assert "NONE" in first_message
    assert "open-graded" in first_message


async def test_grade_retest_answer_transport_failure_raises_transport_failed() -> None:
    """Transport error on open raises kind=transport_failed."""
    transport = FakeTransport(
        [GRADING_RESPONSE_CORRECT],
        raise_on_send=TransportError("network down"),
    )

    with pytest.raises(RetestServiceError) as exc_info:
        await grade_retest_answer(
            transport=transport,
            question="Q",
            expected_answer="A",
            user_answer="A",
        )

    assert exc_info.value.kind == "transport_failed"


async def test_grade_retest_answer_unparseable_response_raises_parse_failed() -> None:
    """LLM returns garbage that the parser can't make sense of."""
    transport = FakeTransport(["this is not a valid grading response"])

    with pytest.raises(RetestServiceError) as exc_info:
        await grade_retest_answer(
            transport=transport,
            question="Q",
            expected_answer="A",
            user_answer="A",
        )

    assert exc_info.value.kind == "parse_failed"


async def test_grade_retest_answer_teaching_turn_rejected_as_wrong_kind() -> None:
    """LLM ignores the intro and returns a teaching turn.

    Falsifying test for the wrong-shape rejection. The retest
    flow has no use for teaching turns from the grader LLM. The
    service must reject and let the caller surface a clear error
    rather than persisting a non-grading turn.
    """
    transport = FakeTransport([TEACHING_TURN_RESPONSE])

    with pytest.raises(RetestServiceError) as exc_info:
        await grade_retest_answer(
            transport=transport,
            question="Q",
            expected_answer="A",
            user_answer="A",
        )

    assert exc_info.value.kind == "wrong_response_kind"


async def test_grade_retest_answer_tool_call_rejected_as_wrong_kind() -> None:
    """Native tool_calls in the grading response are rejected.

    Falsifying test for the tool-call defense. The grading intro
    advertises no tools, but a misbehaving LLM (or one whose
    training data biases toward tool use) might emit one anyway.
    Treat as wrong-shape rather than executing.
    """
    tool_call = GetWeakTopicsCall(args=GetWeakTopicsInput(), id="call_x")
    transport = FakeTransport([TransportResponse(text="", tool_calls=[tool_call])])

    with pytest.raises(RetestServiceError) as exc_info:
        await grade_retest_answer(
            transport=transport,
            question="Q",
            expected_answer="A",
            user_answer="A",
        )

    assert exc_info.value.kind == "wrong_response_kind"


# ---------- answer_retest_question (dispatch path) ----------


async def test_answer_retest_persists_user_and_grading_turns(db: DbSession) -> None:
    """A retest answer persists USER + GRADING turns and commits."""
    topic = _make_topic(db)
    source = _make_source_session(db, topic_id=topic.id)
    _make_learned_item(
        db,
        session_id=source.id,
        topic_id=topic.id,
        question="What is 7 // 2?",
        answer="3",
    )
    db.commit()

    retest, _ = start_retest(
        db=db, source_session_id=source.id, transport_kind=TransportKind.DEEPSEEK
    )

    transport = FakeTransport([GRADING_RESPONSE_CORRECT])
    grading = await answer_retest_question(db=db, transport=transport, session=retest, answer="3")

    assert isinstance(grading, ParsedGrading)
    assert grading.verdict == GradingVerdict.CORRECT

    db.refresh(retest)
    turns = sorted(retest.turns, key=lambda t: t.turn_index)
    # Index 0: synthetic ASSISTANT. Index 1: USER answer. Index 2: GRADING.
    assert len(turns) == 3
    assert turns[1].role is TurnRole.USER
    assert turns[1].raw_content == "3"
    assert turns[2].role is TurnRole.GRADING
    assert turns[2].parsed is not None
    assert turns[2].parsed["verdict"] == "correct"


async def test_answer_retest_grading_failure_does_not_persist_user_turn(
    db: DbSession,
) -> None:
    """Grading failure leaves no USER turn behind.

    Falsifying test for the safe-ordering rule. If we persisted
    USER first then graded, a grading failure would leave an
    orphan USER turn. The current implementation grades first
    and persists after, so a failure means zero new turns are written.
    """
    topic = _make_topic(db)
    source = _make_source_session(db, topic_id=topic.id)
    _make_learned_item(db, session_id=source.id, topic_id=topic.id)
    db.commit()

    retest, _ = start_retest(
        db=db, source_session_id=source.id, transport_kind=TransportKind.DEEPSEEK
    )

    transport = FakeTransport([], raise_on_send=TransportError("boom"))

    with pytest.raises(SessionServiceError):
        await answer_retest_question(db=db, transport=transport, session=retest, answer="3")

    db.refresh(retest)
    turns = sorted(retest.turns, key=lambda t: t.turn_index)
    # Only the synthetic ASSISTANT turn from start_retest survives.
    assert len(turns) == 1
    assert turns[0].role is TurnRole.ASSISTANT


async def test_answer_retest_open_graded_passes_none_to_grader(db: DbSession) -> None:
    """OPEN_ANSWER_PLACEHOLDER items present expected_answer=None.

    Falsifying test for the round-trip: a source LearnedItem with
    answer=OPEN_ANSWER_PLACEHOLDER must produce expected_answer=None
    in the grading call, not the literal placeholder string.
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

    retest, _ = start_retest(
        db=db, source_session_id=source.id, transport_kind=TransportKind.DEEPSEEK
    )

    transport = FakeTransport([GRADING_RESPONSE_OPEN])
    await answer_retest_question(
        db=db, transport=transport, session=retest, answer="my explanation"
    )

    first_prompt = transport.chats[0].messages_sent[1]
    # Should mention NONE (open-graded), not the placeholder.
    assert "NONE" in first_prompt
    assert OPEN_ANSWER_PLACEHOLDER not in first_prompt


# ---------- next_retest_question (dispatch path) ----------


async def test_next_retest_question_persists_next_synthetic_turn(db: DbSession) -> None:
    """Pulling next question persists a synthetic ASSISTANT turn."""
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

    retest, _ = start_retest(
        db=db, source_session_id=source.id, transport_kind=TransportKind.DEEPSEEK
    )
    # Simulate the user answering the first question.
    _add_user_turn(db, session_id=retest.id, turn_index=1)
    # Simulate a grading turn following the user's answer.
    grading_turn = SessionTurn(
        session_id=retest.id,
        turn_index=2,
        role=TurnRole.GRADING,
        raw_content="<grading>",
        parsed={
            "kind": "grading",
            "verdict": "correct",
            "explanation": "Right.",
            "explanation_code": None,
        },
        mode=None,
    )
    db.add(grading_turn)
    db.commit()

    result = await next_retest_question(db=db, session=retest)

    assert isinstance(result, ParsedTurn)
    assert result.question == "Second question"

    db.refresh(retest)
    turns = sorted(retest.turns, key=lambda t: t.turn_index)
    # Index 0: synthetic q1. Index 1: USER. Index 2: GRADING. Index 3: synthetic q2.
    assert len(turns) == 4
    assert turns[3].role is TurnRole.ASSISTANT
    assert turns[3].parsed is not None
    assert turns[3].parsed["question"] == "Second question"


async def test_next_retest_question_returns_session_end_when_exhausted(
    db: DbSession,
) -> None:
    """All source items answered: next question returns ParsedSessionEnd."""
    topic = _make_topic(db)
    source = _make_source_session(db, topic_id=topic.id)
    _make_learned_item(db, session_id=source.id, topic_id=topic.id, question="Only question")
    db.commit()

    retest, _ = start_retest(
        db=db, source_session_id=source.id, transport_kind=TransportKind.DEEPSEEK
    )
    # Simulate the user answering the only question.
    _add_user_turn(db, session_id=retest.id, turn_index=1)
    grading_turn = SessionTurn(
        session_id=retest.id,
        turn_index=2,
        role=TurnRole.GRADING,
        raw_content="<grading>",
        parsed={
            "kind": "grading",
            "verdict": "correct",
            "explanation": "Right.",
            "explanation_code": None,
        },
        mode=None,
    )
    db.add(grading_turn)
    db.commit()

    result = await next_retest_question(db=db, session=retest)

    assert isinstance(result, ParsedSessionEnd)
    assert "1 of 1" in result.summary


# ---------- dispatch via session_service ----------


async def test_send_user_answer_dispatches_retest_path(db: DbSession) -> None:
    """send_user_answer detects parent_session_id and routes to retest path.

    Falsifying test for the dispatch: a session with
    parent_session_id set must NOT call the live-session
    transport methods (resume_chat, send). FakeTransport's
    chats list records every call. The retest path uses
    start_new_chat only, no resume_chat.
    """
    topic = _make_topic(db)
    source = _make_source_session(db, topic_id=topic.id)
    _make_learned_item(db, session_id=source.id, topic_id=topic.id)
    db.commit()

    retest, _ = start_retest(
        db=db, source_session_id=source.id, transport_kind=TransportKind.DEEPSEEK
    )

    transport = FakeTransport([GRADING_RESPONSE_CORRECT])
    result = await send_user_answer(db=db, transport=transport, session_id=retest.id, answer="3")

    assert isinstance(result, ParsedGrading)
    # Retest path opened exactly one fresh chat via start_new_chat.
    # No resume_chat means no chat carries resumed_from metadata.
    assert len(transport.chats) == 1
    assert transport.chats[0].resumed_from is None


async def test_request_next_question_dispatches_retest_path(db: DbSession) -> None:
    """request_next_question detects parent_session_id and routes to retest path.

    The retest path makes no transport calls (questions come from
    source LearnedItems). Falsifying test: assert the transport's
    chat list stays empty.
    """
    topic = _make_topic(db)
    source = _make_source_session(db, topic_id=topic.id)
    _make_learned_item(
        db,
        session_id=source.id,
        topic_id=topic.id,
        question="First",
        created_at=datetime(2026, 5, 1, tzinfo=UTC),
    )
    _make_learned_item(
        db,
        session_id=source.id,
        topic_id=topic.id,
        question="Second",
        created_at=datetime(2026, 5, 2, tzinfo=UTC),
    )
    db.commit()

    retest, _ = start_retest(
        db=db, source_session_id=source.id, transport_kind=TransportKind.DEEPSEEK
    )
    _add_user_turn(db, session_id=retest.id, turn_index=1)
    grading_turn = SessionTurn(
        session_id=retest.id,
        turn_index=2,
        role=TurnRole.GRADING,
        raw_content="<grading>",
        parsed={
            "kind": "grading",
            "verdict": "correct",
            "explanation": "Right.",
            "explanation_code": None,
        },
        mode=None,
    )
    db.add(grading_turn)
    db.commit()

    transport = FakeTransport([])
    result = await request_next_question(db=db, transport=transport, session_id=retest.id)

    assert isinstance(result, ParsedTurn)
    assert result.question == "Second"
    # Retest path uses no transport calls.
    assert len(transport.chats) == 0
