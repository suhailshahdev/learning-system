"""Session HTTP routes.

Three POST endpoints exposing the session-service operations:
start a new session, send a follow-up turn, approve and complete
a session. Routes do error mapping to HTTP status codes and
dispatch to the right transport based on the session's
transport_kind column.
"""

from __future__ import annotations

from typing import Any

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
from app.models import Session, TransportKind
from app.schemas.session_api import (
    ResumeSessionResponse,
    SendTurnRequest,
    SendTurnResponse,
    SessionResponse,
    StartSessionRequest,
    StartSessionResponse,
)
from app.services.parser import ParseError
from app.services.session_resume_service import (
    SessionResumeError,
    get_session_for_resume,
)
from app.services.session_service import (
    SessionServiceError,
    abandon_session,
    approve_session,
    send_user_answer,
    start_session,
)
from app.transport.base import LLMTransport, TransportError

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _map_service_error(exc: SessionServiceError) -> HTTPException:
    """Translate a service-layer error to an HTTP exception.

    Inspects message and cause to pick the right status code:
    not-found is 404, wrong-state is 409, transport or parse
    failures are 502 (the upstream LLM produced something wrong),
    and anything else is 500.
    """
    message = exc.message
    if "not found" in message:
        return HTTPException(status.HTTP_404_NOT_FOUND, detail=message)
    if "expected in_progress" in message:
        return HTTPException(status.HTTP_409_CONFLICT, detail=message)
    if isinstance(exc.cause, (TransportError, ParseError)):
        return HTTPException(status.HTTP_502_BAD_GATEWAY, detail=message)
    return HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=message)


def _pick_transport(
    kind: TransportKind,
    playwright: PlaywrightTransportDep,
    deepseek: DeepseekTransportDep,
) -> LLMTransport[Any]:
    """Dispatch to the matching transport instance.

    Both transports are constructed at app startup and held on
    app.state. The route reads the kind, this function picks.

    Returns LLMTransport[Any] rather than LLMTransport[object]
    because Handle is invariant: a PlaywrightClaudeTransport is
    LLMTransport[PlaywrightChatHandle], not LLMTransport[object].
    Any matches the convention used by the service signatures.
    """
    if kind is TransportKind.CLAUDE_PLAYWRIGHT:
        return playwright
    return deepseek


@router.post(
    "",
    response_model=StartSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def start(
    body: StartSessionRequest,
    db: DbSession,
    playwright: PlaywrightTransportDep,
    deepseek: DeepseekTransportDep,
) -> StartSessionResponse:
    """Open a new session against the chosen transport."""
    transport = _pick_transport(body.transport_kind, playwright, deepseek)

    try:
        session, first_turn = await start_session(
            db=db,
            transport=transport,
            transport_kind=body.transport_kind,
            topic_path=body.topic_path,
        )
    except SessionServiceError as exc:
        raise _map_service_error(exc) from exc

    return StartSessionResponse(
        session=SessionResponse.model_validate(session),
        first_turn=first_turn,
    )


@router.post("/{session_id}/turns", response_model=SendTurnResponse)
async def send_turn(
    session_id: str,
    body: SendTurnRequest,
    db: DbSession,
    playwright: PlaywrightTransportDep,
    deepseek: DeepseekTransportDep,
) -> SendTurnResponse:
    """Send a user answer and return the LLM's parsed reply."""
    session = db.get(Session, session_id)
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Session {session_id!r} not found.")

    transport = _pick_transport(session.transport_kind, playwright, deepseek)

    try:
        parsed = await send_user_answer(
            db=db,
            transport=transport,
            session_id=session_id,
            answer=body.answer,
        )
    except SessionServiceError as exc:
        raise _map_service_error(exc) from exc

    return SendTurnResponse(parsed=parsed)


@router.post("/{session_id}/approve", response_model=SessionResponse)
async def approve(
    session_id: str,
    db: DbSession,
) -> SessionResponse:
    """Approve a completed session, mint learned items."""
    try:
        completed = await approve_session(db=db, session_id=session_id)
    except SessionServiceError as exc:
        raise _map_service_error(exc) from exc

    return SessionResponse.model_validate(completed)


@router.post("/{session_id}/abandon", response_model=SessionResponse)
async def abandon(
    session_id: str,
    db: DbSession,
) -> SessionResponse:
    """Abandon an in-progress session without minting learned items."""
    try:
        abandoned = await abandon_session(db=db, session_id=session_id)
    except SessionServiceError as exc:
        raise _map_service_error(exc) from exc

    return SessionResponse.model_validate(abandoned)


def _map_resume_error(exc: SessionResumeError) -> HTTPException:
    """Translate a resume-service error to an HTTP exception.

    Resume errors carry a kind discriminator so we map per-kind
    rather than substring-matching the message. not_found is 404,
    not_resumable is 409, no_parsed_turn is a data-integrity 500.
    """
    if exc.kind == "not_found":
        return HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc))
    if exc.kind == "not_resumable":
        return HTTPException(status.HTTP_409_CONFLICT, detail=str(exc))
    return HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.get("/{session_id}", response_model=ResumeSessionResponse)
async def resume(
    session_id: str,
    db: DbSession,
) -> ResumeSessionResponse:
    """Cold-load an in-progress session for the frontend.

    Used when the session page loads without route state (deep
    link, refresh, home dashboard click). Returns the session row
    plus the latest parsed assistant response.
    """
    try:
        session_resp, parsed = get_session_for_resume(db=db, session_id=session_id)
    except SessionResumeError as exc:
        raise _map_resume_error(exc) from exc

    return ResumeSessionResponse(session=session_resp, parsed=parsed)
