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

from typing import Literal

from pydantic import BaseModel

# Step kinds. read steps run during planning and feed evidence,
# mutate steps run after approval inside the transaction boundary.
# The split is what lets the orchestrator gather evidence without
# committing anything, then apply mutations atomically once approved.
type StepKind = Literal["read", "mutate"]


class PlanStep(BaseModel):
    """One step in a plan.

    tool names the action to dispatch. args is the validated input
    the orchestrator passes to it. kind decides phase: read steps
    execute during planning, mutate steps wait for approval.

    Args are kept as a free dict for now because there is one
    hardcoded step and no planner validating shapes yet. This will be
    replaced with a discriminated union over the bounded step
    vocabulary so each tool's args are typed at the schema boundary.
    """

    kind: StepKind
    tool: str
    args: dict[str, object]


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
