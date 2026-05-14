"""Schemas for the LLM tool surface.

Six tools let the teaching LLM read and write system state during
a session: list and create domains, list and upsert topics, read
the user's knowledge summary, and read recent sessions. Each tool
has an input schema (what the LLM provides when calling) and an
output schema (what the handler returns after executing).

The tool surface is wire-format-aware. DeepSeek's native function
calling consumes JSON Schema generated from these models. The
Claude transport's structured-prompt fallback embeds these models
in the parser's discriminated union via ParsedToolCall.

Tool schemas are MCP-compatible by convention: JSON Schema
shape, descriptive parameter names, sentinel-free defaults. A
future MCP server wrapper would consume these directly.

Tools are for system actions only. Grading and question
generation stay as structured text output via the existing
ParsedTurn schema. Resist adding tools for those.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 (Pydantic runtime field resolution)
from typing import Annotated, Literal

# Pydantic v2 fails to resolve generic types with TYPE_CHECKING-only
# imports. Same constraint as parsed_response.py.
from app.models.enums import (  # noqa: TC002
    Difficulty,
    DomainKind,
    GradingVerdict,
    LearningMode,
    SessionState,
    TopicStatus,
)
from app.schemas.common import Prerequisite  # noqa: TC002 (Pydantic runtime field resolution)
from pydantic import BaseModel, ConfigDict, Field

# ---------- Tool inputs ----------


class ListDomainsInput(BaseModel):
    """Input for list_domains. No parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class CreateDomainInput(BaseModel):
    """Input for create_domain.

    Idempotent on name: if a domain with this name already exists,
    the handler returns it without modification rather than failing.
    The LLM can call this safely whenever it encounters an unknown
    domain in user input.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=64, description="Unique domain name.")
    kind: DomainKind = Field(description="What sort of domain this is.")
    description: str | None = Field(
        default=None,
        max_length=512,
        description="Short human-readable description, optional.",
    )


class GetTopicsByDomainInput(BaseModel):
    """Input for get_topics_by_domain.

    Returns existing topics in one domain so the LLM can avoid
    re-creating paths that already exist (e.g. 'Python > Data Types
    > Integers' vs 'Python > Datatypes > Int').
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    domain_name: str = Field(min_length=1, max_length=64)


class CreateOrUpdateTopicInput(BaseModel):
    """Input for create_or_update_topic.

    Upsert by path. Difficulty and prerequisites overwrite on
    update only when provided. None means leave existing values
    unchanged. Parent_path is resolved to a topic id at handler
    time.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str = Field(min_length=1, max_length=512, description="Domain > ... > Subtopic.")
    difficulty: Difficulty | None = None
    prerequisites: list[Prerequisite] = Field(default_factory=list)
    parent_path: str | None = Field(default=None, max_length=512)


class GetUserKnowledgeSummaryInput(BaseModel):
    """Input for get_user_knowledge_summary. No parameters."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class GetRecentSessionsInput(BaseModel):
    """Input for get_recent_sessions.

    Limit is bounded so the LLM cannot ask for unreasonable
    history dumps. Default of 5 matches the home dashboard's
    recent-sessions count.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    limit: int = Field(default=5, ge=1, le=20)


class GetWeakTopicsInput(BaseModel):
    """Input for get_weak_topics.

    Returns topics where the user has incorrect or partial grading
    verdicts on past learned items. Used by diagnostic mode to
    surface "where am I failing." The min_attempts floor filters
    out topics with too little data to draw a conclusion.

    sample_size caps how many representative wrong-answer questions
    come back per topic. Three is the working default: enough for
    the LLM to spot patterns, few enough not to bloat responses.
    Set to 0 to get counts only.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_attempts: int = Field(default=2, ge=1, le=50)
    sample_size: int = Field(default=3, ge=0, le=10)


class GetStaleTopicsInput(BaseModel):
    """Input for get_stale_topics.

    Returns topics with old last_reviewed_at timestamps, ordered
    oldest-first. Used by diagnostic mode to surface "what have
    I forgotten." The days_threshold filter keeps the LLM focused
    on topics actually stale enough to warrant revisiting.

    limit caps the result set so the LLM does not get an overwhelming
    list when the user has many old topics.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    days_threshold: int = Field(default=14, ge=1, le=365)
    limit: int = Field(default=10, ge=1, le=50)


# ---------- Tool outputs ----------


class DomainInfo(BaseModel):
    """One domain row in tool output."""

    model_config = ConfigDict(frozen=True)

    name: str
    kind: DomainKind
    description: str | None


class ListDomainsOutput(BaseModel):
    """Output for list_domains."""

    model_config = ConfigDict(frozen=True)

    domains: list[DomainInfo]


class CreateDomainOutput(BaseModel):
    """Output for create_domain.

    `created` is False when the domain already existed. The existing
    row is returned unchanged.
    """

    model_config = ConfigDict(frozen=True)

    created: bool
    domain: DomainInfo


class TopicInfo(BaseModel):
    """One topic row in tool output.

    Slimmer than the home dashboard's TopicSummary: tool consumers
    don't need parent_id, tags, or last_reviewed_at.
    """

    model_config = ConfigDict(frozen=True)

    path: str
    difficulty: Difficulty | None
    status: TopicStatus
    prerequisites: list[Prerequisite]


class GetTopicsByDomainOutput(BaseModel):
    """Output for get_topics_by_domain."""

    model_config = ConfigDict(frozen=True)

    domain: str
    topics: list[TopicInfo]


class CreateOrUpdateTopicOutput(BaseModel):
    """Output for create_or_update_topic.

    `created` is True when a new row was inserted, False when an
    existing row was updated (or left as-is when input fields were
    None).
    """

    model_config = ConfigDict(frozen=True)

    created: bool
    topic: TopicInfo


class KnowledgeRow(BaseModel):
    """One row in the user's knowledge summary.

    Mirrors home_service.KnowledgeSummaryRow but lives here because
    the tool surface is the canonical home for shared shapes. The
    home dashboard consumes from this same shape going forward.
    """

    model_config = ConfigDict(frozen=True)

    domain: str
    difficulty: Difficulty
    count: int


class GetUserKnowledgeSummaryOutput(BaseModel):
    """Output for get_user_knowledge_summary."""

    model_config = ConfigDict(frozen=True)

    rows: list[KnowledgeRow]


class RecentSessionInfo(BaseModel):
    """One recent session in tool output.

    Slimmer than RecentSessionSummary used by the home dashboard:
    tool consumers don't need session id, transport kind, or
    updated_at. Topic_path is the most useful field for the LLM
    deciding whether to continue prior work.
    """

    model_config = ConfigDict(frozen=True)

    topic_path: str | None
    state: SessionState
    mode_used: LearningMode | None
    created_at: datetime


class WrongAnswerSample(BaseModel):
    """One representative wrong-answer question for a weak topic.

    Question text is truncated to 200 chars to give the LLM
    concrete signal without pulling full question history into the
    response. Verdict is included so the LLM can distinguish "tried
    and got it wrong" from "tried and got close."
    """

    model_config = ConfigDict(frozen=True)

    question: str = Field(max_length=200)
    verdict: GradingVerdict


class WeakTopicInfo(BaseModel):
    """One topic where the user has shown weakness.

    Counts are by grading verdict so the LLM can weigh "consistently
    wrong" against "mostly correct, one slip." samples is up to
    sample_size representative wrong-answer questions, or empty if
    the caller passed sample_size=0.
    """

    model_config = ConfigDict(frozen=True)

    topic_path: str
    incorrect_count: int
    partial_count: int
    correct_count: int
    samples: list[WrongAnswerSample]


class GetWeakTopicsOutput(BaseModel):
    """Output for get_weak_topics.

    Topics are ordered worst-first by a simple weakness score
    (incorrect + 0.5 * partial, divided by total). Empty list when
    no topic clears min_attempts.
    """

    model_config = ConfigDict(frozen=True)

    topics: list[WeakTopicInfo]


class StaleTopicInfo(BaseModel):
    """One topic the user has not reviewed recently.

    days_since_review is computed at handler time from
    last_reviewed_at, included pre-computed so the LLM does not have
    to parse and subtract dates.
    """

    model_config = ConfigDict(frozen=True)

    topic_path: str
    last_reviewed_at: datetime
    days_since_review: int


class GetStaleTopicsOutput(BaseModel):
    """Output for get_stale_topics.

    Topics are ordered oldest-first (most stale first). Empty list
    when no topic is older than days_threshold.
    """

    model_config = ConfigDict(frozen=True)

    topics: list[StaleTopicInfo]


class GetRecentSessionsOutput(BaseModel):
    """Output for get_recent_sessions."""

    model_config = ConfigDict(frozen=True)

    sessions: list[RecentSessionInfo]


# ---------- Tool name registry ----------

# Literal type of all valid tool names. Used as the discriminator
# on ParsedToolCall and as the key type for the handler registry.
type ToolName = Literal[
    "list_domains",
    "create_domain",
    "get_topics_by_domain",
    "create_or_update_topic",
    "get_user_knowledge_summary",
    "get_recent_sessions",
    "get_weak_topics",
    "get_stale_topics",
]


# ---------- Discriminated tool-call envelope ----------

# One ToolCall value carries the tool name and validated input.
# Used by the parser when decoding a ---TOOL_CALL--- block on
# the Claude transport, and by the DeepSeek transport when
# normalizing native tool_call entries from the API response.


class ListDomainsCall(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: Literal["list_domains"] = "list_domains"
    args: ListDomainsInput
    # Correlation id from the LLM provider (DeepSeek's native function
    # calling). None for transports without a per-call id concept
    # (Claude via claude.ai's chat wire format). The session-service
    # helper falls back to the tool name when id is None.
    id: str | None = None


class CreateDomainCall(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: Literal["create_domain"] = "create_domain"
    args: CreateDomainInput
    id: str | None = None


class GetTopicsByDomainCall(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: Literal["get_topics_by_domain"] = "get_topics_by_domain"
    args: GetTopicsByDomainInput
    id: str | None = None


class CreateOrUpdateTopicCall(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: Literal["create_or_update_topic"] = "create_or_update_topic"
    args: CreateOrUpdateTopicInput
    id: str | None = None


class GetUserKnowledgeSummaryCall(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: Literal["get_user_knowledge_summary"] = "get_user_knowledge_summary"
    args: GetUserKnowledgeSummaryInput
    id: str | None = None


class GetRecentSessionsCall(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: Literal["get_recent_sessions"] = "get_recent_sessions"
    args: GetRecentSessionsInput
    id: str | None = None


class GetWeakTopicsCall(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: Literal["get_weak_topics"] = "get_weak_topics"
    args: GetWeakTopicsInput
    id: str | None = None


class GetStaleTopicsCall(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: Literal["get_stale_topics"] = "get_stale_topics"
    args: GetStaleTopicsInput
    id: str | None = None


type ToolCall = Annotated[
    ListDomainsCall
    | CreateDomainCall
    | GetTopicsByDomainCall
    | CreateOrUpdateTopicCall
    | GetUserKnowledgeSummaryCall
    | GetRecentSessionsCall
    | GetWeakTopicsCall
    | GetStaleTopicsCall,
    Field(discriminator="name"),
]
