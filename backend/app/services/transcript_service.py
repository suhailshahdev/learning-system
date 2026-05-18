"""Transcript service for the read-only transcript view.

Walks a session's turns and returns the user-visible conversation:
teaching turns, user answers, grading responses, and the optional
session-end proposal. Strips service-noise turns (system prompts,
chat-transition markers, tool plumbing, handover scaffolding).

Read-only. No transport calls, no commits.

The service is the only thing that knows the "what counts as
user-visible" filter rules. Routes pass through and the frontend
renders. Adding a new turn role (e.g. a future evaluator-agent
turn) means deciding here whether it counts as visible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import select

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.orm import Session as DbSession

from app.models import Session, SessionState, SessionTurn, TurnRole
from app.schemas.parsed_response import (
    ParsedGrading,
    ParsedSessionEnd,
    ParsedTurn,
)
from app.schemas.session_api import SessionResponse
from app.schemas.transcript_api import (
    GradingEntry,
    SessionEndEntry,
    TranscriptEntry,
    TranscriptResponse,
    TurnEntry,
    UserAnswerEntry,
)

# Sessions in these states have a meaningful transcript. IN_PROGRESS
# is excluded: the live session page is the right surface for an
# active session and transcript-vs-live duplication would confuse
# the user. The route maps not_eligible to HTTP 409.
_TRANSCRIPT_ELIGIBLE_STATES = frozenset(
    {SessionState.COMPLETED, SessionState.ABANDONED, SessionState.ARCHIVED}
)


class TranscriptServiceError(Exception):
    """Raised when a transcript cannot be produced.

    Carries a kind discriminator so the route layer maps to HTTP
    status codes without string-matching the message. Same pattern
    as SessionResumeError.
    """

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def get_transcript(db: DbSession, session_id: str) -> TranscriptResponse:
    """Build the user-visible transcript of a finished session.

    Raises TranscriptServiceError with kind:
    - "not_found" when session_id is unknown
    - "not_eligible" when session is IN_PROGRESS (live-session
      surface owns that case)
    - "malformed_parsed" when a stored parsed JSON blob fails
      re-validation (data-integrity error)
    """
    session = db.get(Session, session_id)
    if session is None:
        raise TranscriptServiceError("not_found", f"Session {session_id} not found")
    if session.state not in _TRANSCRIPT_ELIGIBLE_STATES:
        raise TranscriptServiceError(
            "not_eligible",
            f"Session {session_id} is in state {session.state.value}; "
            "transcript is available for completed, abandoned, or archived sessions",
        )

    turns = (
        db.execute(
            select(SessionTurn)
            .where(SessionTurn.session_id == session_id)
            .order_by(SessionTurn.turn_index.asc())
        )
        .scalars()
        .all()
    )

    entries = _build_entries(turns)

    return TranscriptResponse(
        session=SessionResponse.model_validate(session),
        entries=entries,
    )


def _build_entries(turns: Sequence[SessionTurn]) -> list[TranscriptEntry]:
    """Walk turns in order, emit one entry per visible turn.

    Keeps a flag for whether the immediately previous emitted entry
    was a teaching turn. USER turns count as visible only when they
    follow a teaching turn (their content is the user's answer).
    USER turns elsewhere (continue prompts after grading) are
    service-generated noise and dropped.

    "Immediately previous emitted entry" lets the pairing skip over
    intervening tool/transition turns: a teaching turn at index 4
    followed by tool turns at 5 and 6 and the user answer at 7
    still pairs correctly.
    """
    entries: list[TranscriptEntry] = []
    last_emitted_was_teaching = False

    for turn in turns:
        entry = _entry_from_turn(turn, last_emitted_was_teaching)
        if entry is None:
            continue
        entries.append(entry)
        last_emitted_was_teaching = isinstance(entry, TurnEntry)

    return entries


def _entry_from_turn(turn: SessionTurn, last_emitted_was_teaching: bool) -> TranscriptEntry | None:
    """Translate one SessionTurn into a TranscriptEntry, or None to drop.

    Branches:
    - ASSISTANT with parsed.kind == "turn" → TurnEntry
    - ASSISTANT with parsed.kind == "session_end" → SessionEndEntry
    - GRADING with parsed.kind == "grading" → GradingEntry
    - USER following a teaching turn → UserAnswerEntry
    - Anything else → None

    SYSTEM turns, TRANSITION turns, TOOL_CALL/TOOL_RESULT turns,
    handover-marker ASSISTANT turns, and service-generated continue-
    prompt USER turns all fall through to None.

    Defense in depth: parsed JSON is re-validated via Pydantic on
    every entry. A malformed blob raises TranscriptServiceError so
    the caller can surface a clear error rather than returning
    corrupted data.
    """
    if turn.role is TurnRole.ASSISTANT:
        return _assistant_entry(turn)
    if turn.role is TurnRole.GRADING:
        return _grading_entry(turn)
    if turn.role is TurnRole.USER and last_emitted_was_teaching:
        return UserAnswerEntry(turn_index=turn.turn_index, answer=turn.raw_content)
    # SYSTEM, TRANSITION, TOOL_CALL, TOOL_RESULT, continue-prompt
    # USER turns: all dropped.
    return None


def _assistant_entry(turn: SessionTurn) -> TranscriptEntry | None:
    """Translate an ASSISTANT turn into TurnEntry, SessionEndEntry, or None.

    ASSISTANT turns carry three parsed kinds in practice: "turn"
    (teaching turn), "session_end" (proposal), and "handover"
    (The marker turn persisted by _continue_with_handover. This
    is internal service noise and is dropped from the visible
    turn list).
    """
    if turn.parsed is None:
        return None
    kind = turn.parsed.get("kind")
    if kind == "turn":
        try:
            parsed = ParsedTurn.model_validate(turn.parsed)
        except Exception as exc:
            raise TranscriptServiceError(
                "malformed_parsed",
                f"Turn {turn.turn_index} has malformed ParsedTurn JSON",
            ) from exc
        return TurnEntry(turn_index=turn.turn_index, turn=parsed)
    if kind == "session_end":
        try:
            parsed_end = ParsedSessionEnd.model_validate(turn.parsed)
        except Exception as exc:
            raise TranscriptServiceError(
                "malformed_parsed",
                f"Turn {turn.turn_index} has malformed ParsedSessionEnd JSON",
            ) from exc
        return SessionEndEntry(turn_index=turn.turn_index, session_end=parsed_end)
    # kind == "handover" or anything else: drop.
    return None


def _grading_entry(turn: SessionTurn) -> GradingEntry | None:
    """Translate a GRADING turn into GradingEntry, or None if parsed is missing."""
    if turn.parsed is None:
        return None
    try:
        parsed_grading = ParsedGrading.model_validate(turn.parsed)
    except Exception as exc:
        raise TranscriptServiceError(
            "malformed_parsed",
            f"Turn {turn.turn_index} has malformed ParsedGrading JSON",
        ) from exc
    return GradingEntry(turn_index=turn.turn_index, grading=parsed_grading)
