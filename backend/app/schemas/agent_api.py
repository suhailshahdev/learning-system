"""Request and response schemas for the agent propose/approve endpoints.

Unlike the other *_api modules, the plan and evidence travel as the
service-layer models rather than field-by-field projections. The
backend holds no state between propose and approve: the client must
send the pair back verbatim, and a projected wire shape would need a
lossless translation in both directions to guarantee that.
"""

from __future__ import annotations

# Pydantic v2 fails to resolve field types imported under
# TYPE_CHECKING. TransportKind, Plan, and Evidence are all field
# annotations, so they must be runtime imports. Same constraint as
# diagnose_api.py and agent_plan.py.
from app.models import TransportKind  # noqa: TC002
from app.schemas.agent_plan import Evidence, Plan  # noqa: TC002
from pydantic import BaseModel, ConfigDict


class AgentProposeRequest(BaseModel):
    """Request body for POST /api/agent/propose.

    Single field: which transport drives the planning chat. No
    session_id because the assistant has no session row.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    transport_kind: TransportKind


class AgentProposeResponse(BaseModel):
    """Response body for POST /api/agent/propose.

    The proposed plan and the evidence that grounds it. The client
    shows the pair to the user and posts it back unchanged on
    approval.
    """

    model_config = ConfigDict(frozen=True)

    plan: Plan
    evidence: list[Evidence]


class AgentApproveRequest(BaseModel):
    """Request body for POST /api/agent/approve.

    The proposal exactly as propose returned it. Groundedness is
    re-checked server-side against this evidence before any step
    executes, so a tampered or stale plan is rejected, not applied.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    plan: Plan
    evidence: list[Evidence]
