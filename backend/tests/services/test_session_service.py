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
    LearningMode,
    Session,
    SessionState,
    SessionTurn,
    Topic,
    TopicStatus,
    TurnRole,
)
from app.services.parser import ParseError
from app.services.session_service import (
    SessionServiceError,
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


async def test_start_session_persists_session_and_turns(db: DbSession) -> None:
    """Happy path: returns parsed turn and writes session + 2 turns."""
    transport = FakeTransport(responses=[VALID_TURN_RESPONSE])

    session, parsed = await start_session(
        db=db,
        transport=transport,
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


async def test_start_session_creates_topic_if_missing(db: DbSession) -> None:
    """A topic at the given path is created when it does not already exist."""
    assert db.query(Topic).count() == 0

    transport = FakeTransport(responses=[VALID_TURN_RESPONSE])
    await start_session(
        db=db,
        transport=transport,
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
        topic_path="Python > Data Types > Integers",
    )

    assert db.query(Topic).count() == 1
    assert session.topic_id == existing.id


async def test_start_session_raises_on_parse_failure(db: DbSession) -> None:
    """Garbage from the LLM raises SessionServiceError; nothing is committed."""
    transport = FakeTransport(responses=["not a delimited response"])

    with pytest.raises(SessionServiceError) as exc:
        await start_session(
            db=db,
            transport=transport,
            topic_path="Python > Data Types > Integers",
        )

    assert isinstance(exc.value.cause, ParseError)
    assert db.query(Session).count() == 0
    assert db.query(SessionTurn).count() == 0


async def test_start_session_raises_on_transport_error(db: DbSession) -> None:
    """A transport failure raises SessionServiceError; nothing is committed."""
    transport = FakeTransport(
        responses=[],
        raise_on_send=TransportError("simulated network error"),
    )

    with pytest.raises(SessionServiceError) as exc:
        await start_session(
            db=db,
            transport=transport,
            topic_path="Python > Data Types > Integers",
        )

    assert isinstance(exc.value.cause, TransportError)
    assert db.query(Session).count() == 0
    assert db.query(SessionTurn).count() == 0


async def test_start_session_raises_on_wrong_response_kind(db: DbSession) -> None:
    """A SESSION_END_PROPOSAL on session start is an LLM bug; raises."""
    transport = FakeTransport(responses=[SESSION_END_RESPONSE])

    with pytest.raises(SessionServiceError, match="Expected a teaching turn"):
        await start_session(
            db=db,
            transport=transport,
            topic_path="Python > Data Types > Integers",
        )

    assert db.query(Session).count() == 0
    assert db.query(SessionTurn).count() == 0
