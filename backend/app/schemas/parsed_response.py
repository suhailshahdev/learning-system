"""Schemas for parsed LLM responses.

The transport layer returns raw text. The parser turns that text
into one of three structured shapes: a regular teaching turn, a
session-end proposal, or a handover block emitted at the chat-
length threshold. The session engine consumes these and persists
the right rows.
"""

from __future__ import annotations

from typing import Annotated, Literal

# Pydantic v2 stores field types as ForwardRef when only available
# under TYPE_CHECKING; bare references happen to resolve through
# late-binding lookup, but generic-wrapped types (e.g. list[Difficulty])
# fail with PydanticUserError. Runtime import is the only pattern
# that works for both shapes. Same constraint as SQLAlchemy
# `Mapped[T]` columns (handover D75); ruff's TC002 doesn't account
# for runtime-introspecting libraries.
from app.models.enums import Difficulty, LearningMode  # noqa: TC002
from pydantic import BaseModel, ConfigDict, Field


class Prerequisite(BaseModel):
    """One prerequisite from the PREREQUISITES block.

    The wire format is `topic_path:difficulty` per pair, comma-
    separated. The parser splits and validates, so consumers see
    structured data and can run prereq checks without re-parsing
    strings.
    """

    model_config = ConfigDict(frozen=True)

    topic_path: str = Field(min_length=1)
    min_difficulty: Difficulty


class ParsedTurn(BaseModel):
    """A regular teaching turn.

    The wire format has TOPIC, DIFFICULTY, PREREQUISITES, MODE,
    QUESTION, EXPECTED_ANSWER, REQUIREMENTS, FOLLOWUP, TAGS. Three
    of those fields can be absent in a meaningful way: EXPECTED_ANSWER
    can be the literal "OPEN" for free-form modes, and REQUIREMENTS /
    FOLLOWUP can be the literal "NONE". The parser converts those
    sentinels to Python None so consumers do not branch on strings.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["turn"] = "turn"
    topic_path: str = Field(min_length=1)
    difficulty: Difficulty
    prerequisites: list[Prerequisite] = Field(default_factory=list)
    mode: LearningMode
    question: str = Field(min_length=1)
    expected_answer: str | None = None
    requirements: str | None = None
    followup: str | None = None
    tags: list[str] = Field(default_factory=list)


class ParsedSessionEnd(BaseModel):
    """The LLM proposed the session is complete.

    User confirms via the approve button to mark items learned;
    user can also keep going if they disagree. The summary is the
    one-line text the LLM emits between SESSION_END_PROPOSAL and END.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["session_end"] = "session_end"
    summary: str = Field(min_length=1)


class ParsedHandover(BaseModel):
    """Handover block emitted when a chat hits the message-count threshold.

    The session engine pastes this into the next chat as part of
    its intro. The fields are kept as raw strings because the next
    chat is the only consumer; further structuring would have no use.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["handover"] = "handover"
    domain_focus: str
    covered: str
    last_question: str
    next_planned: str
    open_threads: str
    user_state: str


# Discriminated union over the three response shapes. Pydantic
# narrows on the `kind` field; mypy follows along on match-case
# blocks. Use this as the parser's return type so consumers can
# pattern-match exhaustively.
type ParsedResponse = Annotated[
    ParsedTurn | ParsedSessionEnd | ParsedHandover,
    Field(discriminator="kind"),
]
