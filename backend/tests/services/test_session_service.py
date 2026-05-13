"""Tests for the session service.

Uses the FakeTransport from tests/services/fakes.py so tests
exercise real service logic against real DB writes without
hitting any LLM. Each test gets a fresh in-memory SQLite database
via the conftest fixture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from app.models import (
    AssertionSource,
    Difficulty,
    Domain,
    DomainKind,
    ErrorLog,
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
    UserKnowledgeAssertion,
)
from app.schemas.parsed_response import ParsedGrading, ParsedTurn
from app.services.parser import ParseError
from app.services.prereq_service import PrereqsUnmetError
from app.services.session_service import (
    HANDOVER_THRESHOLD,
    OPEN_ANSWER_PLACEHOLDER,
    SessionServiceError,
    abandon_session,
    approve_session,
    request_next_question,
    send_user_answer,
    start_session,
)
from app.transport.base import TransportError, TransportResponse

from tests.services.fakes import FakeTransport

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession


VALID_TURN_RESPONSE = """\
---TOPIC---
Python > Data Types > Integers
---DIFFICULTY---
beginner
---PREREQUISITES---
NONE
---MODE---
flashcard
---GRADING---
NONE
---GRADING_EXPLANATION---
NONE
---GRADING_EXPLANATION_CODE---
NONE
---QUESTION---
What is the result of 7 // 2 in Python 3?
---QUESTION_CODE---
NONE
---EXPECTED_ANSWER---
3
---REQUIREMENTS---
NONE
---FOLLOWUP---
NONE
---TAGS---
arithmetic, integer-division
---END---
"""


SESSION_END_RESPONSE = """\
---SESSION_END_PROPOSAL---
Covered the basics of Python integer division.
---END---
"""


PREREQ_TURN_RESPONSE = """\
---TOPIC---
FastAPI > Routing > Path Parameters
---DIFFICULTY---
intermediate
---PREREQUISITES---
Python > Basics:beginner, HTTP & APIs > Methods:beginner
---MODE---
flashcard
---GRADING---
NONE
---GRADING_EXPLANATION---
NONE
---GRADING_EXPLANATION_CODE---
NONE
---QUESTION---
What does the type annotation in a path parameter do?
---QUESTION_CODE---
NONE
---EXPECTED_ANSWER---
It tells FastAPI to convert and validate the value at request time.
---REQUIREMENTS---
NONE
---FOLLOWUP---
NONE
---TAGS---
fastapi, routing
---END---
"""


async def test_start_session_persists_session_and_turns(db: DbSession) -> None:
    """Happy path: returns parsed turn and writes session + 2 turns."""
    transport = FakeTransport(responses=[VALID_TURN_RESPONSE])

    session, parsed = await start_session(
        db=db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        topic_path="Python > Data Types > Integers",
    )

    assert session.state == SessionState.IN_PROGRESS
    assert session.mode_used == LearningMode.FLASHCARD
    assert session.claude_chat_message_count == 1

    assert parsed.topic_path == "Python > Data Types > Integers"
    assert parsed.mode == LearningMode.FLASHCARD

    turns = (
        db.query(SessionTurn)
        .filter(SessionTurn.session_id == session.id)
        .order_by(SessionTurn.turn_index)
        .all()
    )
    assert len(turns) == 2
    assert turns[0].role == TurnRole.SYSTEM
    assert turns[0].turn_index == 0
    assert turns[0].mode is None
    assert turns[1].role == TurnRole.ASSISTANT
    assert turns[1].turn_index == 1
    assert turns[1].mode == LearningMode.FLASHCARD
    assert turns[1].parsed is not None


async def test_start_session_persists_transport_kind(db: DbSession) -> None:
    """The transport_kind passed in is persisted on the session row."""
    transport = FakeTransport(responses=[VALID_TURN_RESPONSE])

    session, _ = await start_session(
        db=db,
        transport=transport,
        transport_kind=TransportKind.CLAUDE_PLAYWRIGHT,
        topic_path="Python > Data Types > Integers",
    )

    assert session.transport_kind == TransportKind.CLAUDE_PLAYWRIGHT


async def test_start_session_creates_topic_if_missing(db: DbSession) -> None:
    """A topic at the given path is created when it does not already exist."""
    assert db.query(Topic).count() == 0

    transport = FakeTransport(responses=[VALID_TURN_RESPONSE])
    await start_session(
        db=db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        topic_path="Python > Data Types > Integers",
    )

    topics = db.query(Topic).all()
    assert len(topics) == 1
    assert topics[0].path == "Python > Data Types > Integers"
    assert topics[0].domain == "Python"
    assert topics[0].name == "Integers"
    assert topics[0].status == TopicStatus.IN_PROGRESS


async def test_start_session_reuses_existing_topic(db: DbSession) -> None:
    """A pre-existing topic at the given path is used, not duplicated."""
    existing = Topic(
        path="Python > Data Types > Integers",
        domain="Python",
        name="Integers",
        status=TopicStatus.NOT_STARTED,
    )
    db.add(existing)
    db.commit()

    transport = FakeTransport(responses=[VALID_TURN_RESPONSE])
    session, _ = await start_session(
        db=db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        topic_path="Python > Data Types > Integers",
    )

    assert db.query(Topic).count() == 1
    assert session.topic_id == existing.id


async def test_start_session_raises_on_parse_failure(db: DbSession) -> None:
    """Garbage from the LLM raises SessionServiceError, nothing is committed."""
    transport = FakeTransport(responses=["not a delimited response"])

    with pytest.raises(SessionServiceError) as exc:
        await start_session(
            db=db,
            transport=transport,
            transport_kind=TransportKind.DEEPSEEK,
            topic_path="Python > Data Types > Integers",
        )

    assert isinstance(exc.value.cause, ParseError)
    assert db.query(Session).count() == 0
    assert db.query(SessionTurn).count() == 0


async def test_start_session_raises_on_transport_error(db: DbSession) -> None:
    """A transport failure raises SessionServiceError and nothing is committed."""
    transport = FakeTransport(
        responses=[],
        raise_on_send=TransportError("simulated network error"),
    )

    with pytest.raises(SessionServiceError) as exc:
        await start_session(
            db=db,
            transport=transport,
            transport_kind=TransportKind.DEEPSEEK,
            topic_path="Python > Data Types > Integers",
        )

    assert isinstance(exc.value.cause, TransportError)
    assert db.query(Session).count() == 0
    assert db.query(SessionTurn).count() == 0


async def test_start_session_raises_on_wrong_response_kind(db: DbSession) -> None:
    """A SESSION_END_PROPOSAL on session start is an LLM bug and raises an error."""
    transport = FakeTransport(responses=[SESSION_END_RESPONSE])

    with pytest.raises(SessionServiceError, match="Expected a teaching turn"):
        await start_session(
            db=db,
            transport=transport,
            transport_kind=TransportKind.DEEPSEEK,
            topic_path="Python > Data Types > Integers",
        )

    assert db.query(Session).count() == 0
    assert db.query(SessionTurn).count() == 0


async def test_start_session_persists_topic_prerequisites(db: DbSession) -> None:
    """The first ParsedTurn's prereqs land on the topic row for later checks."""
    transport = FakeTransport(responses=[PREREQ_TURN_RESPONSE])

    await start_session(
        db=db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        topic_path="FastAPI > Routing > Path Parameters",
    )

    topic = db.query(Topic).filter(Topic.path == "FastAPI > Routing > Path Parameters").one()
    assert topic.prerequisites == [
        {"topic_path": "Python > Basics", "min_difficulty": "beginner"},
        {"topic_path": "HTTP & APIs > Methods", "min_difficulty": "beginner"},
    ]


async def test_start_session_does_not_overwrite_existing_topic_prerequisites(
    db: DbSession,
) -> None:
    """A topic with prereqs already set is not overwritten by later sessions."""
    existing = Topic(
        path="FastAPI > Routing > Path Parameters",
        domain="FastAPI",
        name="Path Parameters",
        status=TopicStatus.IN_PROGRESS,
        prerequisites=[
            {"topic_path": "Python > Basics", "min_difficulty": "advanced"},
        ],
    )
    db.add(existing)
    # Pre-satisfy that prereq so the session start does not raise.
    db.add(
        UserKnowledgeAssertion(
            topic_path="Python > Basics",
            difficulty=Difficulty.ADVANCED,
            source=AssertionSource.SELF_DECLARED,
        )
    )
    db.commit()

    transport = FakeTransport(responses=[PREREQ_TURN_RESPONSE])
    await start_session(
        db=db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        topic_path="FastAPI > Routing > Path Parameters",
    )

    db.refresh(existing)
    # Original prereqs preserved. LLM's new declaration ignored.
    assert existing.prerequisites == [
        {"topic_path": "Python > Basics", "min_difficulty": "advanced"},
    ]


async def test_start_session_raises_on_unmet_prerequisite(db: DbSession) -> None:
    """A topic with stored unmet prereqs raises before any transport call."""
    db.add(
        Topic(
            path="FastAPI > Routing > Path Parameters",
            domain="FastAPI",
            name="Path Parameters",
            status=TopicStatus.NOT_STARTED,
            prerequisites=[
                {"topic_path": "Python > Basics", "min_difficulty": "intermediate"},
            ],
        )
    )
    db.commit()

    transport = FakeTransport(responses=[])  # no responses needed, transport is never called

    with pytest.raises(PrereqsUnmetError) as exc:
        await start_session(
            db=db,
            transport=transport,
            transport_kind=TransportKind.DEEPSEEK,
            topic_path="FastAPI > Routing > Path Parameters",
        )

    assert len(exc.value.unmet) == 1
    assert exc.value.unmet[0].topic_path == "Python > Basics"
    assert exc.value.unmet[0].min_difficulty == Difficulty.INTERMEDIATE
    assert exc.value.unmet[0].asserted_difficulty is None
    # No DB writes from the failed start.
    assert db.query(Session).count() == 0
    assert db.query(SessionTurn).count() == 0


async def test_start_session_proceeds_when_prerequisites_satisfied(db: DbSession) -> None:
    """A topic with all prereqs satisfied starts normally."""
    db.add(
        Topic(
            path="FastAPI > Routing > Path Parameters",
            domain="FastAPI",
            name="Path Parameters",
            status=TopicStatus.NOT_STARTED,
            prerequisites=[
                {"topic_path": "Python > Basics", "min_difficulty": "beginner"},
            ],
        )
    )
    db.add(
        UserKnowledgeAssertion(
            topic_path="Python > Basics",
            difficulty=Difficulty.INTERMEDIATE,
            source=AssertionSource.SELF_DECLARED,
        )
    )
    db.commit()

    transport = FakeTransport(responses=[PREREQ_TURN_RESPONSE])
    session, _ = await start_session(
        db=db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        topic_path="FastAPI > Routing > Path Parameters",
    )

    assert session.state == SessionState.IN_PROGRESS


SECOND_TURN_RESPONSE = """\
---TOPIC---
Python > Data Types > Integers
---DIFFICULTY---
beginner
---PREREQUISITES---
NONE
---MODE---
type_the_answer
---GRADING---
correct
---GRADING_EXPLANATION---
Right. Floor division on positive integers truncates toward zero.
---GRADING_EXPLANATION_CODE---
NONE
---QUESTION---
What is 10 % 3 in Python?
---QUESTION_CODE---
NONE
---EXPECTED_ANSWER---
1
---REQUIREMENTS---
NONE
---FOLLOWUP---
NONE
---TAGS---
arithmetic, modulo
---END---
"""

# A grading response for a correct user answer. Used in tests that
# exercise the post-split send_user_answer flow where the LLM
# replies with grading-only.
GRADING_CORRECT_RESPONSE = """\
---GRADING---
correct
---GRADING_EXPLANATION---
Right. Floor division on positive integers truncates toward zero.
---GRADING_EXPLANATION_CODE---
NONE
---END---
"""

# A grading response for an incorrect user answer.
GRADING_INCORRECT_RESPONSE = """\
---GRADING---
incorrect
---GRADING_EXPLANATION---
Not quite. 7 // 2 evaluates to 3, not 2.5. The // operator is
integer floor division.
---GRADING_EXPLANATION_CODE---
NONE
---END---
"""


async def test_send_user_answer_persists_user_and_grading_turns(db: DbSession) -> None:
    """Happy path: persists user answer + grading at indexes 2 and 3.

    After split, send_user_answer returns a grading response,
    not a teaching turn. The persisted turns reflect that: USER
    answer at index 2, GRADING at index 3 with role=GRADING and
    mode=None (grading turns have no mode).
    """
    transport = FakeTransport(responses=[VALID_TURN_RESPONSE, GRADING_CORRECT_RESPONSE])
    session, _ = await start_session(
        db=db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        topic_path="Python > Data Types > Integers",
    )

    parsed = await send_user_answer(
        db=db,
        transport=transport,
        session_id=session.id,
        answer="3",
    )

    assert isinstance(parsed, ParsedGrading)
    assert parsed.verdict == GradingVerdict.CORRECT

    turns = (
        db.query(SessionTurn)
        .filter(SessionTurn.session_id == session.id)
        .order_by(SessionTurn.turn_index)
        .all()
    )
    assert len(turns) == 4
    assert turns[2].role == TurnRole.USER
    assert turns[2].turn_index == 2
    assert turns[2].raw_content == "3"
    assert turns[2].mode is None
    assert turns[3].role == TurnRole.GRADING
    assert turns[3].turn_index == 3
    assert turns[3].mode is None

    db.refresh(session)
    # mode_used reflects the most recent teaching turn, which is the
    # one from start_session (FLASHCARD). The next request_next_question
    # call will land a teaching turn and update mode_used then.
    assert session.mode_used == LearningMode.FLASHCARD
    assert session.claude_chat_message_count == 2


async def test_send_user_answer_rebuilds_chat_metadata(db: DbSession) -> None:
    """The transport receives metadata built from the persisted turns."""
    transport = FakeTransport(responses=[VALID_TURN_RESPONSE, SECOND_TURN_RESPONSE])
    session, _ = await start_session(
        db=db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        topic_path="Python > Data Types > Integers",
    )

    await send_user_answer(
        db=db,
        transport=transport,
        session_id=session.id,
        answer="3",
    )

    # Two chats produced: one from start_new_chat, one from resume_chat.
    assert len(transport.chats) == 2
    resumed = transport.chats[1]
    assert resumed.resumed_from is not None
    metadata = resumed.resumed_from
    # Resumed metadata should reflect both persisted turns from start.
    assert len(metadata.prior_messages) == 2
    assert metadata.prior_messages[0].role == "system"
    assert metadata.prior_messages[1].role == "assistant"
    assert metadata.message_count == 1


async def test_send_user_answer_rejects_non_in_progress_session(db: DbSession) -> None:
    """A completed session cannot accept new turns."""
    transport = FakeTransport(responses=[VALID_TURN_RESPONSE])
    session, _ = await start_session(
        db=db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        topic_path="Python > Data Types > Integers",
    )
    session.state = SessionState.COMPLETED
    db.commit()

    with pytest.raises(SessionServiceError, match="expected in_progress"):
        await send_user_answer(
            db=db,
            transport=transport,
            session_id=session.id,
            answer="3",
        )


async def test_send_user_answer_rejects_unknown_session(db: DbSession) -> None:
    """An unknown session id raises a clear error."""
    transport = FakeTransport(responses=[])

    with pytest.raises(SessionServiceError, match="not found"):
        await send_user_answer(
            db=db,
            transport=transport,
            session_id="00000000-0000-0000-0000-000000000000",
            answer="3",
        )


async def test_send_user_answer_rolls_back_on_parse_failure(db: DbSession) -> None:
    """Garbage from the LLM rolls back the user turn too."""
    transport = FakeTransport(responses=[VALID_TURN_RESPONSE, "not a delimited response"])
    session, _ = await start_session(
        db=db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        topic_path="Python > Data Types > Integers",
    )
    turns_before = db.query(SessionTurn).count()

    with pytest.raises(SessionServiceError):
        await send_user_answer(
            db=db,
            transport=transport,
            session_id=session.id,
            answer="3",
        )

    # Neither user turn nor assistant turn was written.
    turns_after = db.query(SessionTurn).count()
    assert turns_after == turns_before


OPEN_ANSWER_TURN_RESPONSE = """\
---TOPIC---
Python > Data Types > Integers
---DIFFICULTY---
intermediate
---PREREQUISITES---
NONE
---MODE---
explain_back
---GRADING---
NONE
---GRADING_EXPLANATION---
NONE
---GRADING_EXPLANATION_CODE---
NONE
---QUESTION---
Explain in your own words how Python integers handle arbitrary precision.
---QUESTION_CODE---
NONE
---EXPECTED_ANSWER---
OPEN
---REQUIREMENTS---
NONE
---FOLLOWUP---
NONE
---TAGS---
integers
---END---
"""


async def test_approve_session_mints_learned_items_and_completes(db: DbSession) -> None:
    """Happy path: each Q/A pair becomes a learned_item and session goes COMPLETED."""
    transport = FakeTransport(responses=[VALID_TURN_RESPONSE, SECOND_TURN_RESPONSE])
    session, _ = await start_session(
        db=db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        topic_path="Python > Data Types > Integers",
    )
    await send_user_answer(
        db=db,
        transport=transport,
        session_id=session.id,
        answer="3",
    )

    refreshed = await approve_session(db=db, session_id=session.id)

    assert refreshed.state == SessionState.COMPLETED

    items = (
        db.query(LearnedItem)
        .filter(LearnedItem.session_id == session.id)
        .order_by(LearnedItem.created_at)
        .all()
    )
    assert len(items) == 1
    item = items[0]
    assert item.question == "What is the result of 7 // 2 in Python 3?"
    assert item.answer == "3"
    assert item.your_answer == "3"
    assert item.mode == LearningMode.FLASHCARD
    assert item.status == LearnedItemStatus.LEARNED
    assert item.last_reviewed_at is not None


async def test_approve_session_skips_unanswered_teaching_turn(db: DbSession) -> None:
    """A teaching turn with no following user answer is not minted."""
    transport = FakeTransport(responses=[VALID_TURN_RESPONSE])
    session, _ = await start_session(
        db=db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        topic_path="Python > Data Types > Integers",
    )

    # No send_user_answer call — the first teaching turn has no user answer.
    refreshed = await approve_session(db=db, session_id=session.id)

    assert refreshed.state == SessionState.COMPLETED
    assert db.query(LearnedItem).count() == 0


async def test_approve_session_uses_placeholder_for_open_answer(db: DbSession) -> None:
    """OPEN expected_answer becomes the placeholder string."""
    transport = FakeTransport(responses=[OPEN_ANSWER_TURN_RESPONSE, SECOND_TURN_RESPONSE])
    session, _ = await start_session(
        db=db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        topic_path="Python > Data Types > Integers",
    )
    await send_user_answer(
        db=db,
        transport=transport,
        session_id=session.id,
        answer="They grow as needed without overflow.",
    )

    await approve_session(db=db, session_id=session.id)

    items = db.query(LearnedItem).order_by(LearnedItem.created_at).all()
    assert len(items) == 1
    assert items[0].answer == OPEN_ANSWER_PLACEHOLDER
    assert items[0].your_answer == "They grow as needed without overflow."


async def test_approve_session_skips_session_end_turn(db: DbSession) -> None:
    """A SESSION_END_PROPOSAL turn is not minted as a learned item."""
    transport = FakeTransport(responses=[VALID_TURN_RESPONSE, SESSION_END_RESPONSE])
    session, _ = await start_session(
        db=db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        topic_path="Python > Data Types > Integers",
    )
    await send_user_answer(
        db=db,
        transport=transport,
        session_id=session.id,
        answer="3",
    )

    await approve_session(db=db, session_id=session.id)

    items = db.query(LearnedItem).all()
    # Only the (q1, a1) pair becomes an item. The SESSION_END_PROPOSAL is not paired.
    assert len(items) == 1


async def test_approve_session_resolves_per_item_topic(db: DbSession) -> None:
    """A teaching turn whose topic_path differs from the session's mints under that topic."""
    transport = FakeTransport(responses=[VALID_TURN_RESPONSE, SECOND_TURN_RESPONSE])
    # Start session against a different topic than what the LLM teaches.
    session, _ = await start_session(
        db=db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        topic_path="Python > Overview",
    )
    await send_user_answer(
        db=db,
        transport=transport,
        session_id=session.id,
        answer="3",
    )

    await approve_session(db=db, session_id=session.id)

    items = db.query(LearnedItem).all()
    assert len(items) == 1
    item_topic = db.get(Topic, items[0].topic_id)
    assert item_topic is not None
    # ParsedTurn topic_path was "Python > Data Types > Integers", not "Python > Overview"
    assert item_topic.path == "Python > Data Types > Integers"


async def test_approve_session_rejects_non_in_progress_session(db: DbSession) -> None:
    """A session already completed cannot be approved again."""
    transport = FakeTransport(responses=[VALID_TURN_RESPONSE])
    session, _ = await start_session(
        db=db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        topic_path="Python > Data Types > Integers",
    )
    session.state = SessionState.COMPLETED
    db.commit()

    with pytest.raises(SessionServiceError, match="expected in_progress"):
        await approve_session(db=db, session_id=session.id)


async def test_approve_session_rejects_unknown_session(db: DbSession) -> None:
    """An unknown session id raises a clear error."""
    with pytest.raises(SessionServiceError, match="not found"):
        await approve_session(
            db=db,
            session_id="00000000-0000-0000-0000-000000000000",
        )


async def test_approve_session_runs_derivation_within_transaction(db: DbSession) -> None:
    """Approving a session that minted enough items lands the derived assertion."""
    # Four canned responses: one for start_session, three for follow-ups.
    # Three send_user_answer calls produce three answered (assistant, user)
    # pairs, which mint three learned items at beginner difficulty on the
    # same topic — exactly DERIVATION_THRESHOLD.
    transport = FakeTransport(
        responses=[
            VALID_TURN_RESPONSE,
            SECOND_TURN_RESPONSE,
            VALID_TURN_RESPONSE,
            SECOND_TURN_RESPONSE,
        ]
    )
    session, _ = await start_session(
        db=db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        topic_path="Python > Data Types > Integers",
    )
    await send_user_answer(
        db=db,
        transport=transport,
        session_id=session.id,
        answer="3",
    )
    await send_user_answer(
        db=db,
        transport=transport,
        session_id=session.id,
        answer="1",
    )
    await send_user_answer(
        db=db,
        transport=transport,
        session_id=session.id,
        answer="3",
    )

    await approve_session(db=db, session_id=session.id)

    derived = (
        db.query(UserKnowledgeAssertion)
        .filter(UserKnowledgeAssertion.source == AssertionSource.DERIVED_FROM_LEARNED_ITEMS)
        .all()
    )
    assert len(derived) == 1
    assert derived[0].topic_path == "Python > Data Types > Integers"
    assert derived[0].difficulty == Difficulty.BEGINNER


HANDOVER_RESPONSE = """\
---HANDOVER---
DOMAIN_FOCUS: Python
COVERED: Iterators (intermediate)
LAST_QUESTION: Asked about iter() vs next(), user answered correctly.
NEXT_PLANNED: Generators
OPEN_THREADS: NONE
USER_STATE: Engaged
---END_HANDOVER---
"""


async def test_send_user_answer_below_threshold_uses_within_chat_path(db: DbSession) -> None:
    """A session below the threshold takes the existing path: 2 new turns, no transition."""
    transport = FakeTransport(responses=[VALID_TURN_RESPONSE, SECOND_TURN_RESPONSE])
    session, _ = await start_session(
        db=db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        topic_path="Python > Data Types > Integers",
    )

    # Force the session well below the threshold.
    session.claude_chat_message_count = 1
    db.commit()

    parsed = await send_user_answer(
        db=db,
        transport=transport,
        session_id=session.id,
        answer="3",
    )

    assert isinstance(parsed, ParsedTurn)

    turns = (
        db.query(SessionTurn)
        .filter(SessionTurn.session_id == session.id)
        .order_by(SessionTurn.turn_index)
        .all()
    )
    # 2 turns from start_session + 2 turns from send_user_answer
    assert len(turns) == 4
    transition_turns = [t for t in turns if t.role == TurnRole.TRANSITION]
    assert len(transition_turns) == 0


async def test_abandon_session_marks_state_and_persists(db: DbSession) -> None:
    """Happy path: state goes ABANDONED, no learned items, session row updated."""
    transport = FakeTransport(responses=[VALID_TURN_RESPONSE])
    session, _ = await start_session(
        db=db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        topic_path="Python > Data Types > Integers",
    )

    refreshed = await abandon_session(db=db, session_id=session.id)

    assert refreshed.state == SessionState.ABANDONED
    assert db.query(LearnedItem).count() == 0


async def test_abandon_session_preserves_partial_turns(db: DbSession) -> None:
    """Session turns from the partial session stay intact for replay."""
    transport = FakeTransport(responses=[VALID_TURN_RESPONSE, SECOND_TURN_RESPONSE])
    session, _ = await start_session(
        db=db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        topic_path="Python > Data Types > Integers",
    )
    await send_user_answer(
        db=db,
        transport=transport,
        session_id=session.id,
        answer="3",
    )

    turns_before = db.query(SessionTurn).filter(SessionTurn.session_id == session.id).count()
    await abandon_session(db=db, session_id=session.id)
    turns_after = db.query(SessionTurn).filter(SessionTurn.session_id == session.id).count()

    assert turns_after == turns_before
    assert turns_after == 4


async def test_abandon_session_rejects_unknown_session(db: DbSession) -> None:
    """An unknown session id raises a clear error."""
    with pytest.raises(SessionServiceError, match="not found"):
        await abandon_session(
            db=db,
            session_id="00000000-0000-0000-0000-000000000000",
        )


async def test_abandon_session_rejects_completed_session(db: DbSession) -> None:
    """A session already completed cannot be abandoned."""
    transport = FakeTransport(responses=[VALID_TURN_RESPONSE])
    session, _ = await start_session(
        db=db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        topic_path="Python > Data Types > Integers",
    )
    session.state = SessionState.COMPLETED
    db.commit()

    with pytest.raises(SessionServiceError, match="expected in_progress"):
        await abandon_session(db=db, session_id=session.id)


async def test_abandon_session_rejects_already_abandoned(db: DbSession) -> None:
    """A second abandon call on an already-abandoned session raises wrong-state."""
    transport = FakeTransport(responses=[VALID_TURN_RESPONSE])
    session, _ = await start_session(
        db=db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        topic_path="Python > Data Types > Integers",
    )

    await abandon_session(db=db, session_id=session.id)

    with pytest.raises(SessionServiceError, match="expected in_progress"):
        await abandon_session(db=db, session_id=session.id)


# ---------- Tool-execution loop tests ----------


def _list_domains_tool_call_response() -> TransportResponse:
    """A TransportResponse carrying a list_domains TOOL_CALL block.

    Both transports route their tool calls through ToolCall after
    parsing (Claude) or normalizing (DeepSeek native). This fixture
    produces what the parser would yield from a Claude-style block,
    by including the same JSON the parser would validate.
    """
    body = '{"name": "list_domains", "args": {}}'
    return TransportResponse(
        text=f"---TOOL_CALL---\n{body}\n---END---\n",
    )


def _create_domain_tool_call_response(domain_name: str = "GraphQL") -> TransportResponse:
    """A TransportResponse carrying a create_domain TOOL_CALL block."""
    body = f'{{"name": "create_domain", "args": {{"name": "{domain_name}", "kind": "framework"}}}}'
    return TransportResponse(
        text=f"---TOOL_CALL---\n{body}\n---END---\n",
    )


def _bad_create_domain_tool_call_response() -> TransportResponse:
    """A TOOL_CALL response whose handler will fail.

    parent_path references a topic that does not exist, which
    create_or_update_topic rejects with ToolHandlerError.
    """
    body = (
        '{"name": "create_or_update_topic", '
        '"args": {"path": "Python > Bogus > Path", '
        '"parent_path": "Nonexistent > Topic"}}'
    )
    return TransportResponse(
        text=f"---TOOL_CALL---\n{body}\n---END---\n",
    )


async def test_start_session_with_tool_call_executes_handler_and_proceeds(
    db: DbSession,
) -> None:
    """A tool call on session start runs the handler and continues to the teaching turn.

    Verifies the helper drives the loop end-to-end: tool executes,
    handler commits real DB state, transport receives the result,
    the next response parses as a teaching turn, session lands
    in IN_PROGRESS.

    Tool turns are not persisted on session start (session_id=None
    in the helper) per the pre-session-row design call.
    """
    transport = FakeTransport(
        responses=[
            _create_domain_tool_call_response("GraphQL"),
            VALID_TURN_RESPONSE,
        ]
    )

    session, parsed = await start_session(
        db=db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        topic_path="Python > Data Types > Integers",
    )

    # Real handler effect: the GraphQL domain row exists in the DB.
    # This is the falsifying test: not "did execute_tool_call get called"
    # but "did the row land".
    graphql = db.query(Domain).filter(Domain.name == "GraphQL").one_or_none()
    assert graphql is not None
    assert graphql.kind == DomainKind.FRAMEWORK

    # Transport saw the tool result on the chat.
    assert len(transport.chats) == 1
    chat = transport.chats[0]
    assert len(chat.tool_results_received) == 1
    assert len(chat.tool_results_received[0]) == 1

    # Session reached the teaching turn.
    assert session.state == SessionState.IN_PROGRESS
    assert isinstance(parsed, ParsedTurn)
    assert parsed.mode == LearningMode.FLASHCARD

    # Tool turns not persisted on session start.
    tool_turns = (
        db.query(SessionTurn)
        .filter(SessionTurn.session_id == session.id)
        .filter(SessionTurn.role.in_([TurnRole.TOOL_CALL, TurnRole.TOOL_RESULT]))
        .all()
    )
    assert len(tool_turns) == 0


async def test_send_user_answer_with_single_tool_call_persists_turn_pair(
    db: DbSession,
) -> None:
    """A tool call during a follow-up turn persists TOOL_CALL + TOOL_RESULT in order.

    Turn order is: USER (next_index), TOOL_CALL (next_index + 1),
    TOOL_RESULT (next_index + 2), ASSISTANT (next_index + 3). The
    helper's index threading keeps turn_index unbroken.
    """
    transport = FakeTransport(
        responses=[
            VALID_TURN_RESPONSE,
            _list_domains_tool_call_response(),
            SECOND_TURN_RESPONSE,
        ]
    )
    session, _ = await start_session(
        db=db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        topic_path="Python > Data Types > Integers",
    )

    await send_user_answer(
        db=db,
        transport=transport,
        session_id=session.id,
        answer="3",
    )

    turns = (
        db.query(SessionTurn)
        .filter(SessionTurn.session_id == session.id)
        .order_by(SessionTurn.turn_index)
        .all()
    )
    # 2 from start + 4 from send (user + tool_call + tool_result + assistant)
    assert len(turns) == 6
    assert [t.role for t in turns] == [
        TurnRole.SYSTEM,
        TurnRole.ASSISTANT,
        TurnRole.USER,
        TurnRole.TOOL_CALL,
        TurnRole.TOOL_RESULT,
        TurnRole.ASSISTANT,
    ]
    # turn_index is unbroken
    assert [t.turn_index for t in turns] == [0, 1, 2, 3, 4, 5]

    # TOOL_CALL turn has the validated call in parsed
    tool_call_turn = turns[3]
    assert tool_call_turn.parsed is not None
    assert tool_call_turn.parsed["kind"] == "tool_call"
    assert tool_call_turn.parsed["call"]["name"] == "list_domains"

    # TOOL_RESULT turn has the handler's output as structured JSON
    tool_result_turn = turns[4]
    assert tool_result_turn.parsed is not None
    assert "domains" in tool_result_turn.parsed


async def test_send_user_answer_with_chained_tool_calls_persists_all_pairs(
    db: DbSession,
) -> None:
    """Two tool calls in sequence persist two TOOL_CALL + TOOL_RESULT pairs.

    The LLM chains: first list_domains (read), then create_domain
    (write), then produces the teaching turn. Verifies the helper
    loops correctly and indexes thread through.
    """
    transport = FakeTransport(
        responses=[
            VALID_TURN_RESPONSE,
            _list_domains_tool_call_response(),
            _create_domain_tool_call_response("Rust"),
            SECOND_TURN_RESPONSE,
        ]
    )
    session, _ = await start_session(
        db=db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        topic_path="Python > Data Types > Integers",
    )

    await send_user_answer(
        db=db,
        transport=transport,
        session_id=session.id,
        answer="3",
    )

    turns = (
        db.query(SessionTurn)
        .filter(SessionTurn.session_id == session.id)
        .order_by(SessionTurn.turn_index)
        .all()
    )
    # 2 from start + 6 from send (user + 2 tool pairs + assistant)
    assert len(turns) == 8
    assert [t.role for t in turns] == [
        TurnRole.SYSTEM,
        TurnRole.ASSISTANT,
        TurnRole.USER,
        TurnRole.TOOL_CALL,
        TurnRole.TOOL_RESULT,
        TurnRole.TOOL_CALL,
        TurnRole.TOOL_RESULT,
        TurnRole.ASSISTANT,
    ]
    assert [t.turn_index for t in turns] == [0, 1, 2, 3, 4, 5, 6, 7]

    # Verify both handlers actually ran by checking DB state
    rust = db.query(Domain).filter(Domain.name == "Rust").one_or_none()
    assert rust is not None

    # Transport received two separate tool_result batches
    assert len(transport.chats) == 2  # one from start, one resumed
    resumed = transport.chats[1]
    assert len(resumed.tool_results_received) == 2


async def test_send_user_answer_tool_handler_failure_rolls_back_and_logs(
    db: DbSession,
) -> None:
    """A failing tool handler rolls back service writes and logs to error_log.

    Critical falsifying test: verifies the rollback by
    checking the user turn (added before the helper ran) is gone,
    AND verifies the error_log entry has the right kind.
    """
    transport = FakeTransport(
        responses=[
            VALID_TURN_RESPONSE,
            _bad_create_domain_tool_call_response(),
        ]
    )
    session, _ = await start_session(
        db=db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        topic_path="Python > Data Types > Integers",
    )
    turns_before = db.query(SessionTurn).filter(SessionTurn.session_id == session.id).count()

    with pytest.raises(SessionServiceError, match="Tool handler"):
        await send_user_answer(
            db=db,
            transport=transport,
            session_id=session.id,
            answer="3",
        )

    # Rollback: no new turns persisted from the failed send.
    turns_after = db.query(SessionTurn).filter(SessionTurn.session_id == session.id).count()
    assert turns_after == turns_before

    # error_log has the new row with the right kind
    error_logs = (
        db.query(ErrorLog).filter(ErrorLog.kind == "session.tool_call.handler_failed").all()
    )
    assert len(error_logs) == 1
    assert error_logs[0].session_id == session.id
    assert "create_or_update_topic" in error_logs[0].context["tool_name"]


async def test_send_user_answer_tool_calls_increment_message_count(
    db: DbSession,
) -> None:
    """Tool calls advance the chat's message_count toward HANDOVER_THRESHOLD.

    Tool turns count against the per-chat budget.
    FakeTransport.send_tool_results increments message_count, so the
    session row's claude_chat_message_count reflects both the user-
    answer send and any tool-result sends.
    """
    transport = FakeTransport(
        responses=[
            VALID_TURN_RESPONSE,
            _list_domains_tool_call_response(),
            _create_domain_tool_call_response("Go"),
            SECOND_TURN_RESPONSE,
        ]
    )
    session, _ = await start_session(
        db=db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        topic_path="Python > Data Types > Integers",
    )

    # After start: message_count = 1 (the first message)
    assert session.claude_chat_message_count == 1

    await send_user_answer(
        db=db,
        transport=transport,
        session_id=session.id,
        answer="3",
    )

    db.refresh(session)
    # FakeTransport.resume_chat preserves the prior count (1), then
    # the user-answer send adds 1, then two tool-result sends add 2.
    # Total = 4.
    assert session.claude_chat_message_count == 4


# ============================================================
# request_next_question tests
# ============================================================

# A teaching response for the next-question half of the cycle.
# Same shape as SECOND_TURN_RESPONSE but separately named for
# clarity in tests that exercise request_next_question.
NEXT_TURN_RESPONSE = SECOND_TURN_RESPONSE


async def test_request_next_question_persists_continue_and_teaching_turns(db: DbSession) -> None:
    """Happy path: continue prompt and teaching turn persist after grading."""
    transport = FakeTransport(
        responses=[
            VALID_TURN_RESPONSE,
            GRADING_CORRECT_RESPONSE,
            NEXT_TURN_RESPONSE,
        ]
    )
    session, _ = await start_session(
        db=db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        topic_path="Python > Data Types > Integers",
    )
    await send_user_answer(
        db=db,
        transport=transport,
        session_id=session.id,
        answer="3",
    )

    parsed = await request_next_question(
        db=db,
        transport=transport,
        session_id=session.id,
    )

    assert isinstance(parsed, ParsedTurn)

    turns = (
        db.query(SessionTurn)
        .filter(SessionTurn.session_id == session.id)
        .order_by(SessionTurn.turn_index)
        .all()
    )
    # Turns: SYSTEM(0), ASSISTANT(1), USER(2), GRADING(3), USER(4), ASSISTANT(5)
    assert len(turns) == 6
    assert turns[4].role == TurnRole.USER
    assert "Continue with the next teaching turn" in turns[4].raw_content
    assert turns[5].role == TurnRole.ASSISTANT
    assert turns[5].mode == parsed.mode


async def test_request_next_question_rejects_unknown_session(db: DbSession) -> None:
    transport = FakeTransport(responses=[])
    with pytest.raises(SessionServiceError, match="not found"):
        await request_next_question(
            db=db,
            transport=transport,
            session_id="00000000-0000-0000-0000-000000000000",
        )


async def test_request_next_question_rejects_non_in_progress_session(db: DbSession) -> None:
    transport = FakeTransport(responses=[VALID_TURN_RESPONSE])
    session, _ = await start_session(
        db=db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        topic_path="Python > Data Types > Integers",
    )
    session.state = SessionState.COMPLETED
    db.commit()

    with pytest.raises(SessionServiceError, match="expected in_progress"):
        await request_next_question(
            db=db,
            transport=transport,
            session_id=session.id,
        )


async def test_request_next_question_rejects_when_last_turn_is_not_grading(db: DbSession) -> None:
    """Cannot continue when the last turn is a teaching turn (mid-cycle would be wrong shape).

    State guard: request_next_question expects the session to be in
    the post-grading position. If the last turn is ASSISTANT
    (teaching) instead of GRADING, the caller is out of sequence.
    """
    transport = FakeTransport(responses=[VALID_TURN_RESPONSE])
    session, _ = await start_session(
        db=db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        topic_path="Python > Data Types > Integers",
    )
    # Session right after start_session has SYSTEM(0) and ASSISTANT(1).
    # No user answer or grading yet. request_next_question should reject.

    with pytest.raises(SessionServiceError, match="not awaiting a continue"):
        await request_next_question(
            db=db,
            transport=transport,
            session_id=session.id,
        )


async def test_request_next_question_at_threshold_triggers_handover(db: DbSession) -> None:
    """At threshold: handover request, new chat, 5 transition turns persisted."""
    transport = FakeTransport(
        responses=[
            VALID_TURN_RESPONSE,
            GRADING_CORRECT_RESPONSE,
            HANDOVER_RESPONSE,  # dying chat's handover
            NEXT_TURN_RESPONSE,  # new chat's first response (a teaching turn)
        ]
    )
    session, _ = await start_session(
        db=db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        topic_path="Python > Data Types > Integers",
    )
    await send_user_answer(
        db=db,
        transport=transport,
        session_id=session.id,
        answer="3",
    )

    session.claude_chat_message_count = HANDOVER_THRESHOLD
    db.commit()

    parsed = await request_next_question(
        db=db,
        transport=transport,
        session_id=session.id,
    )

    assert isinstance(parsed, ParsedTurn)

    turns = (
        db.query(SessionTurn)
        .filter(SessionTurn.session_id == session.id)
        .order_by(SessionTurn.turn_index)
        .all()
    )
    # Turns: SYSTEM(0), ASSISTANT(1), USER(2), GRADING(3), then 5
    # transition turns from the handover: SYSTEM(4), ASSISTANT(5),
    # TRANSITION(6), USER(7), ASSISTANT(8).
    assert len(turns) == 9
    assert turns[4].role == TurnRole.SYSTEM
    assert turns[5].role == TurnRole.ASSISTANT
    assert turns[6].role == TurnRole.TRANSITION
    assert turns[7].role == TurnRole.USER
    assert "Continue with the next teaching turn" in turns[7].raw_content
    assert turns[8].role == TurnRole.ASSISTANT


async def test_request_next_question_at_threshold_rolls_back_on_handover_failure(
    db: DbSession,
) -> None:
    """If the handover request itself fails, no new turns persist."""
    transport = FakeTransport(
        responses=[
            VALID_TURN_RESPONSE,
            GRADING_CORRECT_RESPONSE,
        ],
        raise_on_send=TransportError("simulated handover failure"),
        raise_on_send_at=1,
    )
    session, _ = await start_session(
        db=db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        topic_path="Python > Data Types > Integers",
    )
    await send_user_answer(
        db=db,
        transport=transport,
        session_id=session.id,
        answer="3",
    )

    session.claude_chat_message_count = HANDOVER_THRESHOLD
    db.commit()

    turns_before = db.query(SessionTurn).count()

    with pytest.raises(SessionServiceError, match="handover request"):
        await request_next_question(
            db=db,
            transport=transport,
            session_id=session.id,
        )

    turns_after = db.query(SessionTurn).count()
    assert turns_after == turns_before


async def test_request_next_question_unexpected_tool_call_in_handover_path(db: DbSession) -> None:
    """Tool calls on the dying chat's handover response are rejected defensively."""
    transport = FakeTransport(
        responses=[
            VALID_TURN_RESPONSE,
            GRADING_CORRECT_RESPONSE,
            _list_domains_tool_call_response(),  # dying chat returns tool call instead of handover
        ]
    )
    session, _ = await start_session(
        db=db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        topic_path="Python > Data Types > Integers",
    )
    await send_user_answer(
        db=db,
        transport=transport,
        session_id=session.id,
        answer="3",
    )

    session.claude_chat_message_count = HANDOVER_THRESHOLD
    db.commit()

    with pytest.raises(SessionServiceError, match="Unexpected tool call in handover"):
        await request_next_question(
            db=db,
            transport=transport,
            session_id=session.id,
        )
