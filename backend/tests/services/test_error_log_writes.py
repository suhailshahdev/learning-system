"""Tests that session-service catch sites write to error_log.

Each test forces one failure path and checks that exactly one
ErrorLog row exists with the expected kind, message, session id,
and context fields. The service-layer tests cover the
raise-and-rollback contract while this file covers the logging
contract specifically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from app.models import (
    ErrorLog,
    Session,
    SessionTurn,
    TransportKind,
)
from app.services.session_service import (
    HANDOVER_THRESHOLD,
    SessionServiceError,
    request_next_question,
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
---GRADING_EXPLANATION_CODE---
NONE
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
arithmetic
---END---
"""


SESSION_END_RESPONSE = """\
---SESSION_END_PROPOSAL---
Done with integers.
---END---
"""


HANDOVER_RESPONSE = """\
---HANDOVER---
DOMAIN_FOCUS: Python
COVERED: Iterators (intermediate)
LAST_QUESTION: Asked about iter() vs next().
NEXT_PLANNED: Generators
OPEN_THREADS: NONE
USER_STATE: Engaged
---END_HANDOVER---
"""

GRADING_CORRECT_RESPONSE = """\
---GRADING---
correct
---GRADING_EXPLANATION---
Right. Floor division on positive integers truncates toward zero.
---GRADING_EXPLANATION_CODE---
NONE
---END---
"""


# ---------------- start_session catch sites ----------------


async def test_start_session_transport_failure_writes_log(db: DbSession) -> None:
    """Transport failure during session start logs one row with expected fields."""
    transport = FakeTransport(
        responses=[],
        raise_on_send=TransportError("simulated network error"),
    )

    with pytest.raises(SessionServiceError):
        await start_session(
            db=db,
            transport=transport,
            transport_kind=TransportKind.DEEPSEEK,
            topic_path="Python > Data Types > Integers",
        )

    rows = db.query(ErrorLog).all()
    assert len(rows) == 1
    row = rows[0]
    assert row.kind == "session.start.transport_failed"
    assert row.session_id is None  # session row not created yet at failure point
    assert row.context["transport_kind"] == "deepseek"
    assert row.context["topic_path"] == "Python > Data Types > Integers"
    assert "simulated network error" in row.message


async def test_start_session_parse_failure_writes_log(db: DbSession) -> None:
    """Parse failure on first response logs one row with raw_response captured."""
    transport = FakeTransport(responses=["not a delimited response"])

    with pytest.raises(SessionServiceError):
        await start_session(
            db=db,
            transport=transport,
            transport_kind=TransportKind.DEEPSEEK,
            topic_path="Python > Data Types > Integers",
        )

    rows = db.query(ErrorLog).all()
    assert len(rows) == 1
    row = rows[0]
    assert row.kind == "session.start.parse_failed"
    assert row.session_id is None
    assert row.context["raw_response"] == "not a delimited response"
    assert row.context["transport_kind"] == "deepseek"


async def test_start_session_wrong_kind_writes_log(db: DbSession) -> None:
    """A SESSION_END_PROPOSAL on session start is wrong-kind and logs."""
    transport = FakeTransport(responses=[SESSION_END_RESPONSE])

    with pytest.raises(SessionServiceError):
        await start_session(
            db=db,
            transport=transport,
            transport_kind=TransportKind.DEEPSEEK,
            topic_path="Python > Data Types > Integers",
        )

    rows = db.query(ErrorLog).all()
    assert len(rows) == 1
    row = rows[0]
    assert row.kind == "session.start.wrong_response_kind"
    assert row.context["expected_kind"] == "turn"
    assert row.context["actual_kind"] == "session_end"


# ---------------- send_user_answer catch sites ----------------


async def test_send_user_answer_transport_failure_writes_log(db: DbSession) -> None:
    """Transport failure on follow-up turn logs with the session_id set."""
    # raise_on_send_at=0 fails the first send() call after start_session.
    # start_new_chat doesn't bump the send counter, so the first send is
    # the resume+send inside send_user_answer.
    transport = FakeTransport(
        responses=[VALID_TURN_RESPONSE],
        raise_on_send=TransportError("simulated send failure"),
        raise_on_send_at=0,
    )
    session, _ = await start_session(
        db=db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        topic_path="Python > Data Types > Integers",
    )

    with pytest.raises(SessionServiceError):
        await send_user_answer(
            db=db,
            transport=transport,
            session_id=session.id,
            answer="3",
        )

    rows = db.query(ErrorLog).filter(ErrorLog.kind == "session.send.transport_failed").all()
    assert len(rows) == 1
    row = rows[0]
    assert row.session_id == session.id
    assert row.context["transport_kind"] == "deepseek"


async def test_send_user_answer_parse_failure_writes_log(db: DbSession) -> None:
    """Parse failure on follow-up logs with raw_response and session_id."""
    transport = FakeTransport(responses=[VALID_TURN_RESPONSE, "garbage response"])
    session, _ = await start_session(
        db=db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        topic_path="Python > Data Types > Integers",
    )

    with pytest.raises(SessionServiceError):
        await send_user_answer(
            db=db,
            transport=transport,
            session_id=session.id,
            answer="3",
        )

    rows = db.query(ErrorLog).filter(ErrorLog.kind == "session.send.parse_failed").all()
    assert len(rows) == 1
    assert rows[0].session_id == session.id
    assert rows[0].context["raw_response"] == "garbage response"


# ---------------- handover catch sites ----------------


async def test_handover_request_transport_failure_writes_log(db: DbSession) -> None:
    """Transport failure during handover request logs the handover-specific kind.

    Handover lives in request_next_question post-M7.5b, so the test
    drives a full cycle: start_session, send_user_answer (gets
    grading back), then request_next_question at threshold.
    """
    transport = FakeTransport(
        responses=[
            VALID_TURN_RESPONSE,
            GRADING_CORRECT_RESPONSE,
        ],
        raise_on_send=TransportError("simulated handover failure"),
        raise_on_send_at=1,  # fail on the handover-request send
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

    with pytest.raises(SessionServiceError):
        await request_next_question(
            db=db,
            transport=transport,
            session_id=session.id,
        )

    rows = (
        db.query(ErrorLog)
        .filter(ErrorLog.kind == "session.handover.request_transport_failed")
        .all()
    )
    assert len(rows) == 1


async def test_handover_request_parse_failure_writes_log(db: DbSession) -> None:
    """Parse failure on dying chat's handover response logs."""
    transport = FakeTransport(
        responses=[
            VALID_TURN_RESPONSE,
            GRADING_CORRECT_RESPONSE,
            "not a handover",
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

    with pytest.raises(SessionServiceError):
        await request_next_question(
            db=db,
            transport=transport,
            session_id=session.id,
        )

    rows = db.query(ErrorLog).filter(ErrorLog.kind == "session.handover.request_parse_failed").all()
    assert len(rows) == 1


async def test_handover_request_wrong_kind_writes_log(db: DbSession) -> None:
    """A teaching turn instead of a handover from dying chat is wrong-kind."""
    transport = FakeTransport(
        responses=[
            VALID_TURN_RESPONSE,
            GRADING_CORRECT_RESPONSE,
            VALID_TURN_RESPONSE,  # third is wrong kind for handover request
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

    with pytest.raises(SessionServiceError):
        await request_next_question(
            db=db,
            transport=transport,
            session_id=session.id,
        )

    rows = db.query(ErrorLog).filter(ErrorLog.kind == "session.handover.wrong_response_kind").all()
    assert len(rows) == 1


# ---------------- contract assertions ----------------


async def test_failed_start_does_not_write_session_or_turns(db: DbSession) -> None:
    """The error_log row commits but the failed start writes nothing else."""
    transport = FakeTransport(responses=["garbage"])

    with pytest.raises(SessionServiceError):
        await start_session(
            db=db,
            transport=transport,
            transport_kind=TransportKind.DEEPSEEK,
            topic_path="Python > Data Types > Integers",
        )

    # error_log row is committed.
    assert db.query(ErrorLog).count() == 1
    # Session and SessionTurn rows are not.
    assert db.query(Session).count() == 0
    assert db.query(SessionTurn).count() == 0
