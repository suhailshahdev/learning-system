"""Schemas for the sessions browse HTTP API.

GET /api/sessions returns a flat list of session rows with the
fields the browse page needs to render and decide what to do next.
BrowseSessionRow is a distinct projection from RecentSessionSummary:
the browse page renders a learned-item count per row to give the
user a signal for which completed sessions are worth revisiting.

Explicit projection rather than ORM passthrough. Adding internal
columns to Session does not auto-leak into this response shape.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003

# Pydantic v2 resolves field type annotations at validation time.
# Generic-wrapped types fail under TYPE_CHECKING-only imports.
from app.models.enums import (  # noqa: TC002
    LearningMode,
    SessionState,
    TransportKind,
)
from pydantic import BaseModel, ConfigDict


class BrowseSessionRow(BaseModel):
    """One row in the sessions browse list.

    Same identification fields as RecentSessionSummary, with two
    additions: learned_item_count is the number of LearnedItems
    minted on this session (zero for in_progress and abandoned
    sessions, positive for most completed ones). The count gives
    the user a signal for which sessions actually produced lasting
    output without forcing them to click into transcript.

    grading_verdict aggregates land in a later sub-phase once
    retest workflow is in place. The browse-row shape supports
    adding correct_count alongside learned_item_count without
    breaking the API contract.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    topic_id: str | None
    topic_path: str | None
    state: SessionState
    transport_kind: TransportKind
    mode_used: LearningMode
    learned_item_count: int
    created_at: datetime
    updated_at: datetime


class BrowseResponse(BaseModel):
    """Response for GET /api/sessions.

    rows are sorted by created_at desc. The frontend renders this
    as a single page, no pagination cursor for now. limit_reached
    signals whether more sessions exist past the cap so the UI
    can show a "more sessions exist, build All Sessions page"
    hint if the user is near the limit.
    """

    model_config = ConfigDict(frozen=True)

    rows: list[BrowseSessionRow]
    limit_reached: bool
