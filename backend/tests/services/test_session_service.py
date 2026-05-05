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
from app.schemas.parsed_response import ParsedTurn
from app.services.parser import ParseError
from app.services.prereq_service import PrereqsUnmetError
from app.services.session_service import (
    HANDOVER_THRESHOLD,
    OPEN_ANSWER_PLACEHOLDER,
    SessionServiceError,
    abandon_session,
    approve_session,
    send_user_answer,
    start_session,
)
from app.transport.base import TransportError

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
---QUESTION---
What is the result of 7 // 2 in Python 3?
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
---QUESTION---
What does the type annotation in a path parameter do?
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
---QUESTION---
What is 10 % 3 in Python?
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


async def test_send_user_answer_persists_user_and_assistant_turns(db: DbSession) -> None:
    """Happy path: persists user and assistant turns at indexes 2 and 3."""
    transport = FakeTransport(responses=[VALID_TURN_RESPONSE, SECOND_TURN_RESPONSE])
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

    assert isinstance(parsed, ParsedTurn)
    assert parsed.mode == LearningMode.TYPE_THE_ANSWER

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
    assert turns[3].role == TurnRole.ASSISTANT
    assert turns[3].turn_index == 3
    assert turns[3].mode == LearningMode.TYPE_THE_ANSWER

    db.refresh(session)
    assert session.mode_used == LearningMode.TYPE_THE_ANSWER
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
---QUESTION---
Explain in your own words how Python integers handle arbitrary precision.
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


async def test_send_user_answer_at_threshold_triggers_handover(db: DbSession) -> None:
    """At threshold: handover request, new chat, 5 new turns persisted."""
    transport = FakeTransport(
        responses=[
            VALID_TURN_RESPONSE,  # session start
            HANDOVER_RESPONSE,  # dying chat's handover response
            SECOND_TURN_RESPONSE,  # new chat's first response
        ]
    )
    session, _ = await start_session(
        db=db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        topic_path="Python > Data Types > Integers",
    )

    session.claude_chat_message_count = HANDOVER_THRESHOLD
    db.commit()

    parsed = await send_user_answer(
        db=db,
        transport=transport,
        session_id=session.id,
        answer="3",
    )

    assert isinstance(parsed, ParsedTurn)
    assert parsed.mode == LearningMode.TYPE_THE_ANSWER

    turns = (
        db.query(SessionTurn)
        .filter(SessionTurn.session_id == session.id)
        .order_by(SessionTurn.turn_index)
        .all()
    )
    # 2 turns from start_session + 5 turns from the handover-driven send.
    assert len(turns) == 7

    # Verify the new turn shape across the transition: SYSTEM, ASSISTANT,
    # TRANSITION, USER, ASSISTANT in that order.
    new_turns = turns[2:]
    assert [t.role for t in new_turns] == [
        TurnRole.SYSTEM,
        TurnRole.ASSISTANT,
        TurnRole.TRANSITION,
        TurnRole.USER,
        TurnRole.ASSISTANT,
    ]

    transition_turn = new_turns[2]
    assert transition_turn.parsed is not None
    assert transition_turn.parsed["kind"] == "handover"
    assert "---HANDOVER---" in transition_turn.raw_content
    assert "DOMAIN_FOCUS: Python" in transition_turn.raw_content

    # Three chats in total: the original from start_session, the resumed dying
    # chat for the handover request, and the new chat opened post-handover.
    assert len(transport.chats) == 3

    # The session's chat URL and count should now reflect the new chat,
    # not the dying one.
    db.refresh(session)
    assert session.claude_chat_message_count == 1


async def test_send_user_answer_at_threshold_rolls_back_on_handover_failure(
    db: DbSession,
) -> None:
    """If the handover request itself fails, no new turns persist."""
    # raise_on_send_at=0 fails the first send() call after start_session.
    # That first send() is the handover request. The setup leaves the
    # session at threshold so send_user_answer takes the handover path
    # and the failure fires there.
    transport = FakeTransport(
        responses=[VALID_TURN_RESPONSE],
        raise_on_send=TransportError("simulated handover failure"),
        raise_on_send_at=0,
    )
    session, _ = await start_session(
        db=db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        topic_path="Python > Data Types > Integers",
    )

    session.claude_chat_message_count = HANDOVER_THRESHOLD
    db.commit()

    turns_before = db.query(SessionTurn).count()

    with pytest.raises(SessionServiceError, match="handover request"):
        await send_user_answer(
            db=db,
            transport=transport,
            session_id=session.id,
            answer="3",
        )

    turns_after = db.query(SessionTurn).count()
    assert turns_after == turns_before


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
