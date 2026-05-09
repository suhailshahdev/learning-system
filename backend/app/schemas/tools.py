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


class CreateDomainCall(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: Literal["create_domain"] = "create_domain"
    args: CreateDomainInput


class GetTopicsByDomainCall(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: Literal["get_topics_by_domain"] = "get_topics_by_domain"
    args: GetTopicsByDomainInput


class CreateOrUpdateTopicCall(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: Literal["create_or_update_topic"] = "create_or_update_topic"
    args: CreateOrUpdateTopicInput


class GetUserKnowledgeSummaryCall(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: Literal["get_user_knowledge_summary"] = "get_user_knowledge_summary"
    args: GetUserKnowledgeSummaryInput


class GetRecentSessionsCall(BaseModel):
    model_config = ConfigDict(frozen=True)
    name: Literal["get_recent_sessions"] = "get_recent_sessions"
    args: GetRecentSessionsInput


type ToolCall = Annotated[
    ListDomainsCall
    | CreateDomainCall
    | GetTopicsByDomainCall
    | CreateOrUpdateTopicCall
    | GetUserKnowledgeSummaryCall
    | GetRecentSessionsCall,
    Field(discriminator="name"),
]
