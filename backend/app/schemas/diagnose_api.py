"""Request and response schemas for the diagnose endpoint.

API schemas are explicit field-by-field projections.
DiagnoseResponse projects from ParsedProposal rather than passing
the parsed shape through directly: wire shape stays stable even
if ParsedProposal evolves.
"""

from __future__ import annotations

# Pydantic v2 fails to resolve TransportKind as an annotation when
# the import is TYPE_CHECKING-only. Runtime import required for the
# discriminated-field resolution. Same constraint as parsed_response.py.
from app.models import TransportKind  # noqa: TC002
from pydantic import BaseModel, ConfigDict, Field


class DiagnoseRequest(BaseModel):
    """Request body for POST /api/diagnose.

    Single field: which transport drives the diagnostic chat.
    No topic_path because the LLM proposes one; and no session_id
    because no session row is created.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    transport_kind: TransportKind


class DiagnoseResponse(BaseModel):
    """Response body for POST /api/diagnose.

    Projects from ParsedProposal: topic_path and reasoning are
    the two fields the user sees. The frontend modal shows the
    reasoning, the accept button posts topic_path to the existing
    start-session endpoint.
    """

    model_config = ConfigDict(frozen=True)

    topic_path: str = Field(min_length=1)
    reasoning: str = Field(min_length=1)
