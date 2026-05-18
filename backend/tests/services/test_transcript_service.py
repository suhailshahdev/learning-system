"""Tests for the transcript service.

Each test seeds a session with the turns the transcript function
needs to filter. Helpers are private to this file so changes here
do not couple to other test files.
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
from app.schemas.transcript_api import (
    GradingEntry,
    SessionEndEntry,
    TurnEntry,
    UserAnswerEntry,
)
from app.services.transcript_service import (
    TranscriptServiceError,
    get_transcript,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession


def _make_session(
    db: DbSession,
    *,
    topic_id: str | None = None,
    state: SessionState = SessionState.COMPLETED,
) -> Session:
    """Seed a session in the requested state. Default COMPLETED for transcript."""
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
    topic = Topic(path=path, domain=domain, name=name, status=TopicStatus.LEARNED)
    db.add(topic)
    db.flush()
    return topic


def _add_turn(
    db: DbSession,
    *,
    session_id: str,
    turn_index: int,
    role: TurnRole,
    raw_content: str = "<raw>",
    parsed: dict[str, object] | None = None,
    mode: LearningMode | None = None,
) -> SessionTurn:
    """Seed one turn with the given role and parsed payload."""
    turn = SessionTurn(
        session_id=session_id,
        turn_index=turn_index,
        role=role,
        raw_content=raw_content,
        parsed=parsed,
        mode=mode,
    )
    db.add(turn)
    db.flush()
    return turn


def _parsed_turn_payload(
    topic_path: str = "Python > Data Types > Integers",
    question: str = "What is an integer?",
) -> dict[str, object]:
    """Build a valid ParsedTurn JSON blob for storage."""
    return {
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


def _parsed_grading_payload(verdict: str = "correct") -> dict[str, object]:
    """Build a valid ParsedGrading JSON blob for storage."""
    return {
        "kind": "grading",
        "verdict": verdict,
        "explanation": "Right answer.",
        "explanation_code": None,
    }


def _parsed_session_end_payload(summary: str = "Good session, time to wrap.") -> dict[str, object]:
    """Build a valid ParsedSessionEnd JSON blob for storage."""
    return {"kind": "session_end", "summary": summary}


def _parsed_handover_marker_payload() -> dict[str, object]:
    """Build a parsed payload for a handover-marker ASSISTANT turn.

    These turns are persisted by _continue_with_handover and carry
    the ParsedHandover shape in their parsed field. They are
    service noise: the user never saw them.
    """
    return {
        "kind": "handover",
        "domain_focus": "Python",
        "covered": "integers, floats",
        "last_question": "What is a float?",
        "next_planned": "complex numbers",
        "open_threads": "none",
        "user_state": "engaged",
    }


async def test_transcript_filters_to_visible_entries_only(db: DbSession) -> None:
    """Full cycle: teaching turn, user answer, grading, next teaching, end.

    All five entries appear in transcript-index order. SYSTEM and
    TRANSITION turns sprinkled in between are filtered out.
    """
    topic = _make_topic(db)
    session = _make_session(db, topic_id=topic.id)
    # SYSTEM turn at index 0 (intro + first_prompt) — must be filtered.
    _add_turn(
        db,
        session_id=session.id,
        turn_index=0,
        role=TurnRole.SYSTEM,
        raw_content="intro and first prompt",
    )
    # Teaching turn 1.
    _add_turn(
        db,
        session_id=session.id,
        turn_index=1,
        role=TurnRole.ASSISTANT,
        parsed=_parsed_turn_payload(),
        mode=LearningMode.FLASHCARD,
    )
    # User answer.
    _add_turn(
        db,
        session_id=session.id,
        turn_index=2,
        role=TurnRole.USER,
        raw_content="A whole number.",
    )
    # Grading.
    _add_turn(
        db,
        session_id=session.id,
        turn_index=3,
        role=TurnRole.GRADING,
        parsed=_parsed_grading_payload(),
    )
    # Continue-prompt USER turn — must be filtered.
    _add_turn(
        db,
        session_id=session.id,
        turn_index=4,
        role=TurnRole.USER,
        raw_content="Continue with the next question.",
    )
    # Teaching turn 2.
    _add_turn(
        db,
        session_id=session.id,
        turn_index=5,
        role=TurnRole.ASSISTANT,
        parsed=_parsed_turn_payload(
            topic_path="Python > Data Types > Floats",
            question="What is a float?",
        ),
        mode=LearningMode.FLASHCARD,
    )
    # SESSION_END proposal.
    _add_turn(
        db,
        session_id=session.id,
        turn_index=6,
        role=TurnRole.ASSISTANT,
        parsed=_parsed_session_end_payload(),
    )
    db.commit()

    response = get_transcript(db=db, session_id=session.id)

    # Five entries: turn, user_answer, grading, turn, session_end.
    # SYSTEM and continue-prompt USER both filtered.
    assert len(response.entries) == 5
    assert isinstance(response.entries[0], TurnEntry)
    assert response.entries[0].turn_index == 1
    assert isinstance(response.entries[1], UserAnswerEntry)
    assert response.entries[1].turn_index == 2
    assert response.entries[1].answer == "A whole number."
    assert isinstance(response.entries[2], GradingEntry)
    assert response.entries[2].turn_index == 3
    assert isinstance(response.entries[3], TurnEntry)
    assert response.entries[3].turn_index == 5
    assert response.entries[3].turn.topic_path == "Python > Data Types > Floats"
    assert isinstance(response.entries[4], SessionEndEntry)
    assert response.entries[4].turn_index == 6


async def test_transcript_filters_transition_and_handover_marker_turns(db: DbSession) -> None:
    """TRANSITION and handover-marker ASSISTANT turns are filtered.

    Falsifying test for the design call to drop chat-boundary
    markers from the transcript. A session that crossed a chat
    threshold has SYSTEM + handover-marker ASSISTANT + TRANSITION
    + USER (continue) + ASSISTANT (new teaching turn) at the boundary.
    Transcript should show only the teaching turn on the far side.
    """
    topic = _make_topic(db)
    session = _make_session(db, topic_id=topic.id)
    # First teaching turn pre-handover.
    _add_turn(
        db,
        session_id=session.id,
        turn_index=1,
        role=TurnRole.ASSISTANT,
        parsed=_parsed_turn_payload(),
        mode=LearningMode.FLASHCARD,
    )
    _add_turn(
        db,
        session_id=session.id,
        turn_index=2,
        role=TurnRole.USER,
        raw_content="Integer answer.",
    )
    _add_turn(
        db,
        session_id=session.id,
        turn_index=3,
        role=TurnRole.GRADING,
        parsed=_parsed_grading_payload(),
    )
    # Handover sequence (5 turns).
    _add_turn(
        db,
        session_id=session.id,
        turn_index=4,
        role=TurnRole.SYSTEM,
        raw_content="handover request prompt",
    )
    _add_turn(
        db,
        session_id=session.id,
        turn_index=5,
        role=TurnRole.ASSISTANT,
        parsed=_parsed_handover_marker_payload(),
    )
    _add_turn(
        db,
        session_id=session.id,
        turn_index=6,
        role=TurnRole.TRANSITION,
        raw_content="rendered handover block",
        parsed=_parsed_handover_marker_payload(),
    )
    _add_turn(
        db,
        session_id=session.id,
        turn_index=7,
        role=TurnRole.USER,
        raw_content="Continue with the next question.",
    )
    # Teaching turn on far side of handover.
    _add_turn(
        db,
        session_id=session.id,
        turn_index=8,
        role=TurnRole.ASSISTANT,
        parsed=_parsed_turn_payload(
            topic_path="Python > Data Types > Floats",
            question="What is a float?",
        ),
        mode=LearningMode.FLASHCARD,
    )
    db.commit()

    response = get_transcript(db=db, session_id=session.id)

    # Three visible entries: pre-handover teaching turn, user
    # answer, grading and then the post-handover teaching turn.
    # Everything in between (SYSTEM handover request, ASSISTANT
    # handover marker, TRANSITION, continue-prompt USER) is
    # filtered out.
    assert len(response.entries) == 4
    assert isinstance(response.entries[0], TurnEntry)
    assert response.entries[0].turn_index == 1
    assert isinstance(response.entries[1], UserAnswerEntry)
    assert isinstance(response.entries[2], GradingEntry)
    assert isinstance(response.entries[3], TurnEntry)
    assert response.entries[3].turn_index == 8


async def test_transcript_pairs_user_answer_across_intervening_tool_turns(db: DbSession) -> None:
    """USER answer at index 5 pairs with teaching turn at index 1.

    The "immediately previous emitted" rule lets the pairing skip
    over TOOL_CALL and TOOL_RESULT turns between the teaching turn
    and the user's answer. Without this rule, tool-using turns
    would drop their user answer from the transcript.
    """
    topic = _make_topic(db)
    session = _make_session(db, topic_id=topic.id)
    _add_turn(
        db,
        session_id=session.id,
        turn_index=1,
        role=TurnRole.ASSISTANT,
        parsed=_parsed_turn_payload(),
        mode=LearningMode.FLASHCARD,
    )
    # Tool plumbing turns between the teaching turn and the
    # user's answer. Realistic shape when the LLM called a tool
    # before producing the teaching turn (though in practice
    # tool turns come before the teaching turn rather than after,
    # but the test still proves the filter ignores them).
    _add_turn(
        db,
        session_id=session.id,
        turn_index=2,
        role=TurnRole.TOOL_CALL,
        raw_content="<tool call>",
        parsed={"call": {"name": "get_user_knowledge_summary", "arguments": {}}},
    )
    _add_turn(
        db,
        session_id=session.id,
        turn_index=3,
        role=TurnRole.TOOL_RESULT,
        raw_content="<tool result>",
        parsed={"result": "ok"},
    )
    _add_turn(
        db,
        session_id=session.id,
        turn_index=4,
        role=TurnRole.USER,
        raw_content="my answer",
    )
    db.commit()

    response = get_transcript(db=db, session_id=session.id)

    assert len(response.entries) == 2
    assert isinstance(response.entries[0], TurnEntry)
    assert isinstance(response.entries[1], UserAnswerEntry)
    assert response.entries[1].answer == "my answer"


async def test_transcript_empty_for_abandoned_session_with_no_visible_turns(
    db: DbSession,
) -> None:
    """An abandoned session with only SYSTEM turns has empty entries."""
    topic = _make_topic(db)
    session = _make_session(db, topic_id=topic.id, state=SessionState.ABANDONED)
    _add_turn(
        db,
        session_id=session.id,
        turn_index=0,
        role=TurnRole.SYSTEM,
        raw_content="intro",
    )
    db.commit()

    response = get_transcript(db=db, session_id=session.id)

    assert response.entries == []
    assert response.session.state == SessionState.ABANDONED


async def test_transcript_available_for_completed_abandoned_archived(db: DbSession) -> None:
    """Three terminal states all yield a transcript."""
    topic = _make_topic(db)
    for state in [SessionState.COMPLETED, SessionState.ABANDONED, SessionState.ARCHIVED]:
        session = _make_session(db, topic_id=topic.id, state=state)
        _add_turn(
            db,
            session_id=session.id,
            turn_index=1,
            role=TurnRole.ASSISTANT,
            parsed=_parsed_turn_payload(),
            mode=LearningMode.FLASHCARD,
        )
    db.commit()

    # Each session loaded individually rather than asserting on the
    # last one: a regression that broke one state would slip through
    # if we only checked the final session.
    for state in [SessionState.COMPLETED, SessionState.ABANDONED, SessionState.ARCHIVED]:
        sessions = db.query(Session).filter(Session.state == state).all()
        assert len(sessions) == 1
        response = get_transcript(db=db, session_id=sessions[0].id)
        assert response.session.state == state


async def test_transcript_404_for_unknown_session(db: DbSession) -> None:
    """Unknown session id raises not_found."""
    with pytest.raises(TranscriptServiceError) as exc_info:
        get_transcript(db=db, session_id="does-not-exist")

    assert exc_info.value.kind == "not_found"


async def test_transcript_409_for_in_progress_session(db: DbSession) -> None:
    """IN_PROGRESS sessions are not eligible for transcript."""
    topic = _make_topic(db)
    session = _make_session(db, topic_id=topic.id, state=SessionState.IN_PROGRESS)
    _add_turn(
        db,
        session_id=session.id,
        turn_index=1,
        role=TurnRole.ASSISTANT,
        parsed=_parsed_turn_payload(),
        mode=LearningMode.FLASHCARD,
    )
    db.commit()

    with pytest.raises(TranscriptServiceError) as exc_info:
        get_transcript(db=db, session_id=session.id)

    assert exc_info.value.kind == "not_eligible"


async def test_transcript_500_on_malformed_parsed_json(db: DbSession) -> None:
    """An ASSISTANT turn with kind='turn' but missing required fields raises malformed_parsed."""
    topic = _make_topic(db)
    session = _make_session(db, topic_id=topic.id)
    # kind is "turn" so the service tries to validate, but the
    # payload is missing every other required field.
    _add_turn(
        db,
        session_id=session.id,
        turn_index=1,
        role=TurnRole.ASSISTANT,
        parsed={"kind": "turn"},
        mode=LearningMode.FLASHCARD,
    )
    db.commit()

    with pytest.raises(TranscriptServiceError) as exc_info:
        get_transcript(db=db, session_id=session.id)

    assert exc_info.value.kind == "malformed_parsed"
