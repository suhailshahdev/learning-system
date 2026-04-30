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


class SendTurnResponse(BaseModel):
    """Response for POST /sessions/{id}/turns.

    Wraps the parsed response from the LLM. Clients branch directly
    on parsed.kind to handle each response shape.
    """

    model_config = ConfigDict(frozen=True)

    parsed: ParsedResponse
