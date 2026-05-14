"""Schemas for parsed LLM responses.

The transport layer returns raw text. The parser turns that text
into one of three structured shapes: a regular teaching turn, a
session-end proposal, or a handover block emitted at the chat-
length threshold. The session engine consumes these and persists
the right rows.
"""

from __future__ import annotations

from typing import Annotated, Literal

# Pydantic v2 fails to resolve generic types like list[Difficulty]
# when the import is TYPE_CHECKING-only. A runtime import is the
# only pattern that works for both plain and generic-wrapped types.
# Same constraint as SQLAlchemy Mapped columns. ruff's TC002 does
# not account for runtime-introspecting libraries.
from app.models.enums import Difficulty, GradingVerdict, LearningMode  # noqa: TC002
from app.schemas.common import Prerequisite  # noqa: TC002 (Pydantic runtime field resolution)
from app.schemas.tools import ToolCall  # noqa: TC002 (Pydantic runtime field resolution)
from pydantic import BaseModel, ConfigDict, Field


class CodeBlock(BaseModel):
    """A code block embedded in a teaching turn.

    The wire format puts the language tag on the first line and the
    code body on the lines that follow. Both pieces are required but
    the field as a whole is optional via the NONE sentinel.
    """

    model_config = ConfigDict(frozen=True)

    language: str = Field(min_length=1)
    body: str = Field(min_length=1)


class ParsedTurn(BaseModel):
    """A regular teaching turn.

    The wire format includes TOPIC, DIFFICULTY, PREREQUISITES, MODE,
    QUESTION, QUESTION_CODE, EXPECTED_ANSWER, REQUIREMENTS, FOLLOWUP,
    and TAGS. Grading lives on its own standalone ParsedGrading
    response: a teaching turn never carries grading fields after the
    split-roundtrip flow.

    Several fields use sentinels for absence. EXPECTED_ANSWER can be
    OPEN for free-form modes. REQUIREMENTS, FOLLOWUP, and QUESTION_CODE
    can be NONE. The parser converts sentinels to None so consumers
    never branch on strings.

    Code blocks are split out of the prose fields so the frontend can
    render them with a monospace font and language label. Inline code
    stays in the prose with backticks and only block-level code uses
    the CODE field.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["turn"] = "turn"
    topic_path: str = Field(min_length=1)
    difficulty: Difficulty
    prerequisites: list[Prerequisite] = Field(default_factory=list)
    mode: LearningMode
    question: str = Field(min_length=1)
    question_code: CodeBlock | None = None
    expected_answer: str | None = None
    requirements: str | None = None
    followup: str | None = None
    tags: list[str] = Field(default_factory=list)


class ParsedSessionEnd(BaseModel):
    """The LLM proposed the session is complete.

    The user can approve to mark items as learned or keep going if
    they disagree. The summary is the one line of text the LLM emits
    between SESSION_END_PROPOSAL and END.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["session_end"] = "session_end"
    summary: str = Field(min_length=1)


class ParsedGrading(BaseModel):
    """Grading of the user's previous answer, emitted as a standalone response.

    LLM's first response to a user answer is grading-only:
    a verdict, an explanation, and optional explanation code.
    The LLM waits for the next user message (which will be
    the system-generated "Continue with the next question.")
    before emitting a teaching turn.

    Wire format mirrors the standalone ParsedSessionEnd / ParsedHandover
    shape: a top-level delimited block terminated by ---END---.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["grading"] = "grading"
    verdict: GradingVerdict
    explanation: str = Field(min_length=1)
    explanation_code: CodeBlock | None = None


class ParsedHandover(BaseModel):
    """Handover block emitted when a chat hits the message-count threshold.

    The session engine pastes this into the next chat as part of its
    intro. Fields are kept as raw strings since the next chat is the
    only consumer and further structuring would have no use.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["handover"] = "handover"
    domain_focus: str
    covered: str
    last_question: str
    next_planned: str
    open_threads: str
    user_state: str


class ParsedToolCall(BaseModel):
    """The LLM is invoking a tool to read or write system state.

    Used by the Claude transport's structured-prompt fallback:
    the LLM emits a ---TOOL_CALL--- block in chat output, the parser
    validates it into this shape, and the session-service loop
    executes the handler and feeds the result back as the next user
    message.

    The DeepSeek transport does not produce ParsedToolCall via the
    parser. It uses native function calling and converts API
    tool_calls into the same ToolCall value internally, then runs
    through the same registry. Both paths converge on the registry
    so handlers stay transport-agnostic.

    raw_text is the original block content the LLM emitted, kept for
    error_log when validation or execution fails.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["tool_call"] = "tool_call"
    call: ToolCall
    raw_text: str


class ParsedProposal(BaseModel):
    """A topic proposal from the diagnostic LLM.

    Emitted by the throwaway diagnostic chat after the LLM has read
    analytical state via tools. The user sees the topic_path and
    reasoning, then accepts (starts a fresh session on that topic)
    or rejects (back to manual topic entry).

    Two fields by design. confidence was rejected because
    the LLM defaults to "high" when asked, the reasoning field
    carries that signal more honestly.

    topic_path is validated only for non-emptiness here. Whether
    the path exists or matches the Domain > Category > Subtopic
    shape is the service layer's job, not the parser's. Matches
    how ParsedTurn.topic_path is treated.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["proposal"] = "proposal"
    topic_path: str = Field(min_length=1)
    reasoning: str = Field(min_length=1)


# Discriminated union over the five response shapes. Pydantic
# narrows on the kind field and mypy follows in match-case blocks.
# Use this as the parser return type so consumers can pattern-match
# exhaustively.
type ParsedResponse = Annotated[
    ParsedTurn
    | ParsedSessionEnd
    | ParsedHandover
    | ParsedToolCall
    | ParsedGrading
    | ParsedProposal,
    Field(discriminator="kind"),
]
