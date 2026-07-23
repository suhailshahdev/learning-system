"""Specialist finding schemas for the multi-agent assistant.

A specialist is an LLM sub-agent that enriches a plan proposal's
evidence. It consults its tool surface for one plan target and
returns a finding: a short grounding note plus the tool evidence
behind it. Plan execution never goes through a specialist; findings
are read-side material for the proposal.
"""

from __future__ import annotations

from typing import Literal

# Pydantic v2 fails to resolve field types imported under
# TYPE_CHECKING. Evidence is a field type on SpecialistResult, so it
# must be a runtime import. Same constraint as agent_plan.py.
from app.schemas.agent_plan import Evidence  # noqa: TC002 (Pydantic runtime field resolution)
from pydantic import BaseModel, Field

type SpecialistErrorKind = Literal[
    "transport_failed",
    "parse_failed",
    "tool_handler_failed",
    "disallowed_tool",
    "ungrounded",
    "unexpected",
]


class SpecialistFinding(BaseModel):
    """One specialist's grounding note for a single plan target.

    specialist names what produced it, mirroring Evidence.tool.
    topic_path is the plan target the finding is about, filled by the
    service from the hand-off context rather than trusted from LLM
    output. summary is the specialist's note, grounded in the tool
    results it gathered.
    """

    specialist: Literal["retrieval_specialist"] = "retrieval_specialist"
    topic_path: str
    summary: str = Field(min_length=1)


class ParsedFinding(BaseModel):
    """A parsed terminal FINDING response from a specialist flow.

    Deliberately not a ParsedResponse union member, for the same
    reason ParsedPlan is not: the finding terminal is only valid in a
    specialist conversation. raw_text preserves the wire body for
    error_log.
    """

    kind: Literal["finding"] = "finding"
    summary: str
    raw_text: str


class SpecialistResult(BaseModel):
    """What one specialist invocation returns.

    The finding is the LLM's synthesis; evidence holds the raw tool
    results it was synthesized from, one entry per retained tool
    call, same shape the planner's evidence uses. The pair travels
    together so a proposal can show both the note and its sources.
    """

    status: Literal["completed"] = "completed"
    finding: SpecialistFinding
    evidence: list[Evidence]


class SpecialistFailure(BaseModel):
    """A failed specialist invocation retained as proposal evidence.

    The planner degrades per target instead of discarding a valid
    proposal. error_kind preserves the typed failure category while
    message is safe to expose to the client and does not copy the
    underlying exception text.
    """

    status: Literal["failed"] = "failed"
    specialist: Literal["retrieval_specialist"] = "retrieval_specialist"
    topic_path: str
    error_kind: SpecialistErrorKind
    message: str = Field(min_length=1)
