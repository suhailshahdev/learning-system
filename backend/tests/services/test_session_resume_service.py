"""Tests for the session resume service.

Each test seeds a session with the turns the resume function
needs to find. The helpers from test_home_service.py are not
imported because coupling the two test files means home test
changes could break resume tests for unrelated reasons.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest
from app.models import (
    LearningMode,
    Session,
    SessionState,
    SessionTurn,
    Topic,
    TopicStatus,
    TransportKind,
    TurnRole,
)
from app.schemas.parsed_response import ParsedGrading, ParsedTurn
from app.services.session_resume_service import (
    SessionResumeError,
    get_session_for_resume,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession


def _make_session(
    db: DbSession,
    *,
    topic_id: str | None = None,
    state: SessionState = SessionState.IN_PROGRESS,
) -> Session:
    """Seed a session in the requested state."""
    session = Session(
        topic_id=topic_id,
        mode_used=LearningMode.FLASHCARD,
        state=state,
        transport_kind=TransportKind.DEEPSEEK,
        active_preferences=[],
        context_snapshot={},
        created_at=datetime(2026, 5, 1, tzinfo=UTC),
    )
    db.add(session)
    db.flush()
    return session


def _make_topic(db: DbSession, path: str = "Python > Data Types > Integers") -> Topic:
    """Seed a topic for use in turn parsed payloads."""
    domain = path.split(" > ", 1)[0]
    name = path.rsplit(" > ", 1)[-1]
    topic = Topic(path=path, domain=domain, name=name, status=TopicStatus.IN_PROGRESS)
    db.add(topic)
    db.flush()
    return topic


def _make_assistant_turn(
    db: DbSession,
    *,
    session_id: str,
    turn_index: int,
    parsed: dict[str, object] | None,
) -> SessionTurn:
    """Seed an ASSISTANT turn with the given parsed payload."""
    turn = SessionTurn(
        session_id=session_id,
        turn_index=turn_index,
        role=TurnRole.ASSISTANT,
        raw_content="<raw>",
        parsed=parsed,
        mode=LearningMode.FLASHCARD,
    )
    db.add(turn)
    db.flush()
    return turn


def _make_user_turn(db: DbSession, *, session_id: str, turn_index: int) -> SessionTurn:
    """Seed a USER turn (no parsed content)."""
    turn = SessionTurn(
        session_id=session_id,
        turn_index=turn_index,
        role=TurnRole.USER,
        raw_content="user answer",
        parsed=None,
        mode=LearningMode.FLASHCARD,
    )
    db.add(turn)
    db.flush()
    return turn


def _make_grading_turn(
    db: DbSession,
    *,
    session_id: str,
    turn_index: int,
    parsed: dict[str, object] | None = None,
) -> SessionTurn:
    """Seed a GRADING turn with a parsed payload."""
    payload = parsed if parsed is not None else _parsed_grading_payload()
    turn = SessionTurn(
        session_id=session_id,
        turn_index=turn_index,
        role=TurnRole.GRADING,
        raw_content="<raw>",
        parsed=payload,
        mode=None,
    )
    db.add(turn)
    db.flush()
    return turn


def _parsed_turn_payload(topic_path: str = "Python > Data Types > Integers") -> dict[str, object]:
    """Build a valid ParsedTurn JSON blob for storage."""
    return {
        "kind": "turn",
        "topic_path": topic_path,
        "difficulty": "beginner",
        "prerequisites": [],
        "mode": "flashcard",
        "grading_verdict": None,
        "grading_explanation": None,
        "grading_explanation_code": None,
        "question": "What is an integer?",
        "question_code": None,
        "expected_answer": "A whole number.",
        "requirements": None,
        "followup": None,
        "tags": [],
    }


def _parsed_grading_payload(verdict: str = "correct") -> dict[str, object]:
    """Build a valid ParsedGrading JSON blob for storage."""
    return {
        "kind": "grading",
        "verdict": verdict,
        "explanation": "Right answer. Integer division truncates.",
        "explanation_code": None,
    }


async def test_resume_returns_latest_assistant_turn(db: DbSession) -> None:
    """Resume picks the highest-turn-index assistant turn with parsed."""
    topic = _make_topic(db)
    session = _make_session(db, topic_id=topic.id)
    _make_assistant_turn(db, session_id=session.id, turn_index=1, parsed=_parsed_turn_payload())
    _make_user_turn(db, session_id=session.id, turn_index=2)
    _make_assistant_turn(
        db,
        session_id=session.id,
        turn_index=3,
        parsed=_parsed_turn_payload(topic_path="Python > Functions > Closures"),
    )
    db.commit()

    session_resp, parsed = get_session_for_resume(db=db, session_id=session.id)

    assert session_resp.id == session.id
    assert isinstance(parsed, ParsedTurn)
    # Latest assistant turn was the closures one, not the integers one.
    assert parsed.topic_path == "Python > Functions > Closures"


async def test_resume_skips_user_and_system_turns(db: DbSession) -> None:
    """USER and SYSTEM turns are skipped even when they have higher index."""
    topic = _make_topic(db)
    session = _make_session(db, topic_id=topic.id)
    _make_assistant_turn(db, session_id=session.id, turn_index=1, parsed=_parsed_turn_payload())
    _make_user_turn(db, session_id=session.id, turn_index=2)
    db.commit()

    session_resp, parsed = get_session_for_resume(db=db, session_id=session.id)

    assert session_resp.id == session.id
    assert isinstance(parsed, ParsedTurn)


async def test_resume_returns_grading_when_it_is_the_latest_turn(db: DbSession) -> None:
    """Mid-grading reload: latest turn is GRADING, resume returns it.

    Falsifying test for the bug where reload during the post-grading
    pre-continue window returned the previous teaching question
    instead of the grading the user was actually looking at.
    Sequence: ASSISTANT(1), USER(2), GRADING(3). Latest by index
    is the grading turn.
    """
    topic = _make_topic(db)
    session = _make_session(db, topic_id=topic.id)
    _make_assistant_turn(db, session_id=session.id, turn_index=1, parsed=_parsed_turn_payload())
    _make_user_turn(db, session_id=session.id, turn_index=2)
    _make_grading_turn(db, session_id=session.id, turn_index=3)
    db.commit()

    session_resp, parsed = get_session_for_resume(db=db, session_id=session.id)

    assert session_resp.id == session.id
    assert isinstance(parsed, ParsedGrading)
    assert parsed.verdict == "correct"


async def test_resume_returns_next_turn_when_grading_followed_by_next_teaching_turn(
    db: DbSession,
) -> None:
    """Post-continue reload: latest is ASSISTANT, returns the next teaching turn.

    Sequence: ASSISTANT(1), USER(2), GRADING(3), ASSISTANT(4).
    The next teaching turn at index 4 is the latest, not the
    grading at index 3.
    """
    topic = _make_topic(db)
    session = _make_session(db, topic_id=topic.id)
    _make_assistant_turn(db, session_id=session.id, turn_index=1, parsed=_parsed_turn_payload())
    _make_user_turn(db, session_id=session.id, turn_index=2)
    _make_grading_turn(db, session_id=session.id, turn_index=3)
    _make_assistant_turn(
        db,
        session_id=session.id,
        turn_index=4,
        parsed=_parsed_turn_payload(topic_path="Python > Functions > Closures"),
    )
    db.commit()

    session_resp, parsed = get_session_for_resume(db=db, session_id=session.id)

    assert session_resp.id == session.id
    assert isinstance(parsed, ParsedTurn)
    assert parsed.topic_path == "Python > Functions > Closures"


async def test_resume_404_for_unknown_session(db: DbSession) -> None:
    """Unknown session id raises not_found."""
    with pytest.raises(SessionResumeError) as exc_info:
        get_session_for_resume(db=db, session_id="does-not-exist")

    assert exc_info.value.kind == "not_found"


async def test_resume_409_for_completed_session(db: DbSession) -> None:
    """Completed sessions cannot be resumed."""
    topic = _make_topic(db)
    session = _make_session(db, topic_id=topic.id, state=SessionState.COMPLETED)
    _make_assistant_turn(db, session_id=session.id, turn_index=1, parsed=_parsed_turn_payload())
    db.commit()

    with pytest.raises(SessionResumeError) as exc_info:
        get_session_for_resume(db=db, session_id=session.id)

    assert exc_info.value.kind == "not_resumable"


async def test_resume_409_for_abandoned_session(db: DbSession) -> None:
    """Abandoned sessions cannot be resumed."""
    topic = _make_topic(db)
    session = _make_session(db, topic_id=topic.id, state=SessionState.ABANDONED)
    _make_assistant_turn(db, session_id=session.id, turn_index=1, parsed=_parsed_turn_payload())
    db.commit()

    with pytest.raises(SessionResumeError) as exc_info:
        get_session_for_resume(db=db, session_id=session.id)

    assert exc_info.value.kind == "not_resumable"


async def test_resume_500_when_no_assistant_turn(db: DbSession) -> None:
    """A session with no assistant turns is a data-integrity error."""
    topic = _make_topic(db)
    session = _make_session(db, topic_id=topic.id)
    # Only a user turn, no assistant turn at all.
    _make_user_turn(db, session_id=session.id, turn_index=1)
    db.commit()

    with pytest.raises(SessionResumeError) as exc_info:
        get_session_for_resume(db=db, session_id=session.id)

    assert exc_info.value.kind == "no_parsed_turn"


async def test_resume_500_when_assistant_turn_has_null_parsed(db: DbSession) -> None:
    """Assistant turn with parsed=None is also a data-integrity error.

    Should never happen under normal operation since session_service
    only persists assistant turns after successful parse, but the
    column is nullable so the resume service must defend.
    """
    topic = _make_topic(db)
    session = _make_session(db, topic_id=topic.id)
    _make_assistant_turn(db, session_id=session.id, turn_index=1, parsed=None)
    db.commit()

    with pytest.raises(SessionResumeError) as exc_info:
        get_session_for_resume(db=db, session_id=session.id)

    assert exc_info.value.kind == "no_parsed_turn"
