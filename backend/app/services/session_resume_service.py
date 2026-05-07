"""Cold-load resume service for an existing session.

Used by GET /api/sessions/{id} when the frontend lands on the
session page without route state (deep link, browser refresh,
home dashboard click). Walks the session's turns, finds the
most recent assistant turn with parsed content, returns the
session row plus the parsed response.

Read-only. No transport calls, no commits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import select

from app.models import Session, SessionState, SessionTurn, TurnRole
from app.schemas.parsed_response import (
    ParsedHandover,
    ParsedResponse,
    ParsedSessionEnd,
    ParsedTurn,
)
from app.schemas.session_api import SessionResponse

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession


class SessionResumeError(Exception):
    """Raised when a session cannot be resumed via cold load.

    Carries a kind discriminator so the route layer can map to
    appropriate HTTP status codes without string-matching the
    message.
    """

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def get_session_for_resume(
    db: DbSession,
    session_id: str,
) -> tuple[SessionResponse, ParsedResponse]:
    """Load session by id and return latest parsed assistant response.

    Raises SessionResumeError with kind:
    - "not_found" when session_id is unknown
    - "not_resumable" when session is in a terminal state
    - "no_parsed_turn" when no assistant turn has parsed content
      (data integrity issue since a session should always have at
      least the first ASSISTANT turn from start_session)
    """
    session = db.get(Session, session_id)
    if session is None:
        raise SessionResumeError("not_found", f"Session {session_id} not found")

    if session.state != SessionState.IN_PROGRESS:
        raise SessionResumeError(
            "not_resumable",
            f"Session {session_id} is in state {session.state.value}; only "
            "in_progress sessions can be resumed",
        )

    latest_turn = (
        db.execute(
            select(SessionTurn)
            .where(SessionTurn.session_id == session_id)
            .where(SessionTurn.role == TurnRole.ASSISTANT)
            .where(SessionTurn.parsed.is_not(None))
            .order_by(SessionTurn.turn_index.desc())
            .limit(1)
        )
        .scalars()
        .one_or_none()
    )

    if latest_turn is None or latest_turn.parsed is None:
        raise SessionResumeError(
            "no_parsed_turn",
            f"Session {session_id} has no assistant turn with parsed content",
        )

    parsed = _validate_parsed(latest_turn.parsed)
    return SessionResponse.model_validate(session), parsed


def _validate_parsed(raw: dict[str, object]) -> ParsedResponse:
    """Re-validate a stored parsed JSON blob via Pydantic.

    Stored JSON could in theory drift if a future migration
    extended ParsedResponse without backfilling. This is the
    same defensive validation pattern as approve_session in
    session_service.py.
    """
    kind = raw.get("kind")
    if kind == "turn":
        return ParsedTurn.model_validate(raw)
    if kind == "session_end":
        return ParsedSessionEnd.model_validate(raw)
    if kind == "handover":
        return ParsedHandover.model_validate(raw)
    raise SessionResumeError(
        "no_parsed_turn",
        f"Stored parsed JSON has unknown kind: {kind!r}",
    )
