"""Request and response schemas for the session HTTP API.

The schemas here form the contract between the backend routes and
any client (the frontend, the smoke scripts, future API consumers).
Explicit field-by-field models rather than ORM-pass-through means
internal Session columns added later don't auto-leak into the API
surface.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003

# Pydantic v2 resolves field type annotations at validation time.
# Bare types (state: SessionState) would survive TYPE_CHECKING but
# generic-wrapped types (list[X]) would fail with PydanticUserError.
# Importing at runtime avoids the trap entirely (see D126).
from app.models.enums import (  # noqa: TC002
    LearningMode,
    SessionState,
    TransportKind,
)
from app.schemas.parsed_response import ParsedResponse, ParsedTurn  # noqa: TC002
from pydantic import BaseModel, ConfigDict, Field


class StartSessionRequest(BaseModel):
    """Body for POST /sessions."""

    model_config = ConfigDict(frozen=True)

    topic_path: str = Field(min_length=1)
    transport_kind: TransportKind


class SendTurnRequest(BaseModel):
    """Body for POST /sessions/{id}/turns."""

    model_config = ConfigDict(frozen=True)

    answer: str = Field(min_length=1)


class SessionResponse(BaseModel):
    """Public projection of a Session row.

    Only fields the API contract commits to. Internal state added
    to the model later (prereq_check_state, error counters, etc.)
    must be added here explicitly to surface in the API.
    """

    model_config = ConfigDict(frozen=True, from_attributes=True)

    id: str
    topic_id: str | None
    state: SessionState
    transport_kind: TransportKind
    mode_used: LearningMode
    claude_chat_url: str | None
    claude_chat_message_count: int
    created_at: datetime
    updated_at: datetime


class StartSessionResponse(BaseModel):
    """Response for POST /sessions.

    Bundles the new session with the parsed first turn so the
    client gets the question to display in one round-trip.
    """

    model_config = ConfigDict(frozen=True)

    session: SessionResponse
    first_turn: ParsedTurn


class StartRetestRequest(BaseModel):
    """Body for POST /sessions/{source_id}/retest.

    transport_kind is the user's choice for which LLM to use when
    grading fires. Free-form mode questions need an LLM, deterministic
    mode questions never trigger a transport call. The choice is stored
    on the new session row regardless so the retest's transport is
    consistent across all grading events.
    """

    model_config = ConfigDict(frozen=True)

    transport_kind: TransportKind


class StartRetestResponse(BaseModel):
    """Response for POST /sessions/{source_id}/retest.

    Same shape as StartSessionResponse so the frontend can reuse the
    live-session start page for retests. first_turn is reconstructed
    from the source's first LearnedItem, not LLM-generated.
    """

    model_config = ConfigDict(frozen=True)

    session: SessionResponse
    first_turn: ParsedTurn


class ResumeSessionResponse(BaseModel):
    """Response for GET /sessions/{id}.

    Cold-load shape used when the frontend lands on the session
    page without route state. parsed is the union of all three
    response kinds because a resumed session might be mid-turn,
    pending session-end approval, or (rarely) at a handover.
    """

    model_config = ConfigDict(frozen=True)

    session: SessionResponse
    parsed: ParsedResponse


class SendTurnResponse(BaseModel):
    """Response for POST /sessions/{id}/turns.

    Wraps the parsed response from the LLM. After the split-roundtrip
    flow, the normal kind here is "grading". Clients branch on
    parsed.kind to handle the rare "session_end" and "handover"
    shapes that may also surface.
    """

    model_config = ConfigDict(frozen=True)

    parsed: ParsedResponse


class ContinueSessionResponse(BaseModel):
    """Response for POST /sessions/{id}/continue.

    Returned after the client signals Continue past a grading
    response. The normal kind is "turn" (the next teaching turn),
    "session_end" and "handover" remain possible.
    """

    model_config = ConfigDict(frozen=True)

    parsed: ParsedResponse
