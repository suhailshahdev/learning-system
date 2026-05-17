"""Diagnose HTTP route.

One POST endpoint exposing the diagnostic-service operation:
the LLM reads analytical state via tools and proposes one topic
for the user to focus on. The chat is throwaway, no session
row is created. The user accepts (POSTs to the existing
start-session endpoint) or rejects (closes the modal).

Diagnostic endpoint lives at /api/diagnose, not /api/sessions/diagnose,
because it does not create a session resource.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, status

# FastAPI resolves Annotated[...]-aliased dependencies at route
# registration via typing.get_type_hints(), which evaluates the
# annotation strings against the module's runtime namespace. The
# dep aliases must be real imports, not TYPE_CHECKING-only.
from app.api.deps import (  # noqa: TC001
    DbSession,
    DeepseekTransportDep,
    PlaywrightTransportDep,
)
from app.models import TransportKind
from app.schemas.diagnose_api import DiagnoseRequest, DiagnoseResponse
from app.services.diagnostic_service import (
    DiagnosticServiceError,
    propose_topic,
)

if TYPE_CHECKING:
    from app.transport.base import LLMTransport

router = APIRouter(prefix="/diagnose", tags=["diagnose"])


def _map_diagnostic_error(exc: DiagnosticServiceError) -> HTTPException:
    """Translate a diagnostic-service error to an HTTP exception.

    Mirrors sessions.py's _map_resume_error: dispatches on the
    error's kind discriminator, not on message substrings.

    transport_failed, parse_failed, and wrong_response_kind all
    indicate the upstream LLM produced something wrong: 502
    Bad Gateway. tool_handler_failed is a backend issue (the
    handler raised, not the LLM): 500. no_data means the request
    is well-formed but the system has nothing to diagnose:
    422 Unprocessable Entity. unexpected is the catch-all: 500.
    """
    if exc.kind == "no_data":
        return HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=exc.message)
    if exc.kind in ("transport_failed", "parse_failed", "wrong_response_kind"):
        return HTTPException(status.HTTP_502_BAD_GATEWAY, detail=exc.message)
    return HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=exc.message)


def _pick_transport(
    kind: TransportKind,
    playwright: PlaywrightTransportDep,
    deepseek: DeepseekTransportDep,
) -> LLMTransport[Any]:
    """Dispatch to the matching transport instance.

    This is a per-module helper. Duplicated from sessions.py
    because route modules keep small helpers local until a
    third caller emerges. If a future route needs the same
    logic, this lifts to app/api/deps.py.

    Returns LLMTransport[Any]: Handle TypeVar is invariant so
    LLMTransport[object] would not accept the concrete handles.
    """
    if kind is TransportKind.CLAUDE_PLAYWRIGHT:
        return playwright
    return deepseek


@router.post("", response_model=DiagnoseResponse)
async def diagnose(
    body: DiagnoseRequest,
    db: DbSession,
    playwright: PlaywrightTransportDep,
    deepseek: DeepseekTransportDep,
) -> DiagnoseResponse:
    """Run the diagnostic flow and return a topic proposal.

    The LLM reads analytical state via tools (get_weak_topics,
    get_stale_topics, get_topics_by_domain, get_recent_sessions)
    and produces a PROPOSAL block. The chat is closed before this
    handler returns.
    """
    transport = _pick_transport(body.transport_kind, playwright, deepseek)

    try:
        proposal = await propose_topic(
            db=db,
            transport=transport,
            transport_kind=body.transport_kind,
        )
    except DiagnosticServiceError as exc:
        raise _map_diagnostic_error(exc) from exc

    return DiagnoseResponse(
        topic_path=proposal.topic_path,
        reasoning=proposal.reasoning,
    )
