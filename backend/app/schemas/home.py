"""Response schemas for the home dashboard endpoint.

The home dashboard composes six pieces of data: a blank-slate
flag, the most recent in-progress session, items due for review,
in-progress topics grouped by domain, recent sessions, and a
knowledge summary by domain and difficulty.

Schemas are field-by-field projections of the underlying models.
Internal columns added later (prereq state, error counters, etc.)
do not auto-leak into the API surface.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003

# Pydantic v2 resolves field type annotations at validation time.
# Generic-wrapped types (list[X]) fail under TYPE_CHECKING-only
# imports with PydanticUserError. Same constraint as parsed_response.py.
from app.models.enums import (  # noqa: TC002
    Difficulty,
    LearningMode,
    SessionState,
    TopicStatus,
    TransportKind,
)
from app.schemas.session_api import SessionResponse  # noqa: TC002
from pydantic import BaseModel, ConfigDict


class TopicSummary(BaseModel):
    """Compact projection of a Topic row for the home dashboard.

    Excludes parent_id, prerequisites, tags, and timestamps.
    Consumers that need the full topic load it via the topics
    endpoint.
    """

    model_config = ConfigDict(frozen=True, from_attributes=True)

    id: str
    path: str
    name: str
    domain: str
    difficulty: Difficulty | None
    status: TopicStatus


class DomainFocus(BaseModel):
    """In-progress topics grouped under one domain.

    The home screen renders this as a section per domain showing
    what the user is currently working through.
    """

    model_config = ConfigDict(frozen=True)

    domain: str
    in_progress_topics: list[TopicSummary]


class LearnedItemSummary(BaseModel):
    """Compact projection of a LearnedItem row for the review queue.

    Carries enough to render a review card and link to the item's
    detail view: the question, what topic it belongs to, when it
    was last reviewed.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    question: str
    topic_path: str
    difficulty: Difficulty | None
    mode: LearningMode
    last_reviewed_at: datetime | None


class RecentSessionSummary(BaseModel):
    """Compact projection of a Session row for the recent-sessions list.

    Carries topic_path joined from the Topic table so the dashboard
    row can show "Python > Data Types > Integers · in_progress"
    without an N+1 fetch. Drops claude_chat_url and message count
    because both are internal to the live session loop.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    topic_id: str | None
    topic_path: str | None
    state: SessionState
    transport_kind: TransportKind
    mode_used: LearningMode
    created_at: datetime
    updated_at: datetime


class KnowledgeSummaryRow(BaseModel):
    """One row of the knowledge summary table.

    Aggregates user knowledge assertions by domain and difficulty.
    The home screen renders this as "Python: intermediate (12),
    beginner (4); FastAPI: beginner (3)".
    """

    model_config = ConfigDict(frozen=True)

    domain: str
    difficulty: Difficulty
    count: int


class HomeResponse(BaseModel):
    """Composed dashboard payload for GET /api/home.

    is_blank_slate is true when the user has never started a
    session, has no learned items, and has not configured a
    default stack, resume, or target job description. The
    frontend uses this to show the Guided/Direct bootstrap UI.
    """

    model_config = ConfigDict(frozen=True)

    is_blank_slate: bool
    continue_last: SessionResponse | None
    due_for_review: list[LearnedItemSummary]
    focus_by_domain: list[DomainFocus]
    recent_sessions: list[RecentSessionSummary]
    knowledge_summary: list[KnowledgeSummaryRow]
