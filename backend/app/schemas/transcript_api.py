"""Schemas for the transcript HTTP API.

The transcript endpoint returns a list of user-visible entries
in turn-index order, each a member of a discriminated union over
four kinds: teaching turn, user answer, grading response, and
session-end proposal.

ParsedTurn, ParsedGrading, and ParsedSessionEnd from
parsed_response are reused directly as nested fields. The
discriminator field on the entry types (kind) is a literal string
distinct from the discriminator on the nested parsed shapes so
the frontend pattern-matches on entry.kind without unwrapping.
"""

from __future__ import annotations

from typing import Annotated, Literal

# Pydantic v2 resolves field type annotations at validation time.
# Generic-wrapped types fail under TYPE_CHECKING-only imports with
# PydanticUserError. Same constraint as parsed_response.py.
from app.schemas.parsed_response import (  # noqa: TC002
    ParsedGrading,
    ParsedSessionEnd,
    ParsedTurn,
)
from app.schemas.session_api import SessionResponse  # noqa: TC002
from pydantic import BaseModel, ConfigDict, Field


class TurnEntry(BaseModel):
    """A teaching turn the user saw."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["turn"] = "turn"
    turn_index: int
    turn: ParsedTurn


class UserAnswerEntry(BaseModel):
    """The user's answer to the previous teaching turn."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["user_answer"] = "user_answer"
    turn_index: int
    answer: str


class GradingEntry(BaseModel):
    """A grading response from the LLM."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["grading"] = "grading"
    turn_index: int
    grading: ParsedGrading


class SessionEndEntry(BaseModel):
    """A session-end proposal from the LLM.

    Present in transcripts of COMPLETED sessions (user approved
    the proposal). May also appear in ABANDONED sessions if the
    LLM proposed end but the user closed the tab without approving.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["session_end"] = "session_end"
    turn_index: int
    session_end: ParsedSessionEnd


# Discriminated union over the four entry shapes. Pydantic narrows
# on the kind field. The frontend can pattern-match exhaustively.
type TranscriptEntry = Annotated[
    TurnEntry | UserAnswerEntry | GradingEntry | SessionEndEntry,
    Field(discriminator="kind"),
]


class TranscriptResponse(BaseModel):
    """Response for GET /sessions/{id}/transcript.

    Bundles the session row (so the frontend renders header info
    without a second fetch) with the ordered list of visible
    entries. Empty entries list is valid (e.g. abandoned right
    after start, before any user answer).
    """

    model_config = ConfigDict(frozen=True)

    session: SessionResponse
    entries: list[TranscriptEntry]
