"""Plan and step schemas for the multi-agent assistant.

A Plan is an ordered list of PlanSteps the orchestrator walks. Each
step is either a read (runs during planning, produces evidence, never
mutates) or a mutate (runs only after plan-level approval, under the
orchestrator's transaction boundary).

Currently hardcodes a single Plan to exercise the orchestrator's
transaction contract. The bounded planner that produces Plans from a
fixed step vocabulary lands next. These types are deliberately
thin and will grow there.
"""

from __future__ import annotations

from typing import Annotated, Literal

# Pydantic v2 fails to resolve field types imported under
# TYPE_CHECKING. The step args reuse the tool input schemas, so they
# must be runtime imports. Same constraint as parsed_response.py.
from app.schemas.tools import (  # noqa: TC002 (Pydantic runtime field resolution)
    GetWeakTopicsInput,
    MarkForRevisionInput,
)
from pydantic import BaseModel, Field

# Step kinds. read steps run during planning and feed evidence,
# mutate steps run after approval inside the transaction boundary.
# The split is what lets the orchestrator gather evidence without
# committing anything, then apply mutations atomically once approved.
type StepKind = Literal["read", "mutate"]


class GetWeakTopicsStep(BaseModel):
    """A read step that surfaces the user's weak topics.

    kind is pinned to "read" so the orchestrator's phase check
    (step.kind == "read") routes it to the read pass. tool is the
    discriminator: it is unique per step variant, where kind is not
    (every read step shares kind="read").
    """

    kind: Literal["read"] = "read"
    tool: Literal["get_weak_topics"] = "get_weak_topics"
    args: GetWeakTopicsInput


class MarkForRevisionStep(BaseModel):
    """A mutate step that marks one existing topic for revision.

    kind is pinned to "mutate" so it runs only in the approved mutate
    pass, never during planning. The target must already exist: the
    mutate core raises if the path is unknown, and the planner's
    groundedness guard rejects ungrounded targets before approval.
    """

    kind: Literal["mutate"] = "mutate"
    tool: Literal["mark_for_revision"] = "mark_for_revision"
    args: MarkForRevisionInput


# The bounded step vocabulary. The planner composes an ordered
# sequence from this closed set. It cannot invent tool names or pass
# free-form args, because each variant pins its tool and types its
# args to a tool input schema. Pydantic discriminates on tool, and
# mypy follows in match-case blocks. The union grows as specialists
# land; the discriminator stays tool.
type PlanStep = Annotated[
    GetWeakTopicsStep | MarkForRevisionStep,
    Field(discriminator="tool"),
]


class Plan(BaseModel):
    """An ordered list of steps the orchestrator walks.

    steps execute in order. The orchestrator runs every read step
    during planning, then on approval runs every mutate step inside
    one transaction. Mixed ordering is allowed in the type but the
    current flow runs reads first. Ordering rules are decided by the
    planner.
    """

    steps: list[PlanStep]


class Evidence(BaseModel):
    """One read step's output, retained as justification for the plan.

    tool names what produced it, result is the read's output. The
    orchestrator collects one Evidence per read step during planning
    and returns them with the plan so a proposal can show why it was
    made. This is the substrate for linked-evidence UI.
    """

    tool: str
    result: dict[str, object]


class ParsedPlan(BaseModel):
    """A parsed terminal PLAN response from the planner flow.

    Deliberately not a ParsedResponse union member: the plan terminal
    is only valid in the planner conversation, and keeping it off the
    union means the teaching and diagnostic flows cannot parse one by
    accident. raw_text preserves the wire body for error_log.
    """

    kind: Literal["plan"] = "plan"
    plan: Plan
    raw_text: str


class PlanProposal(BaseModel):
    """A proposed plan with the evidence that grounds it.

    What propose returns and what approve receives back. The pair
    travels together because the backend keeps no state between the
    two calls: the groundedness guard re-checks the plan against this
    evidence on approval.
    """

    plan: Plan
    evidence: list[Evidence]
