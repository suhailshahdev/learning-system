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
from app.models.enums import SessionState  # noqa: TC001
from app.schemas.browse_api import BrowseResponse
from app.schemas.session_api import (
    ContinueSessionResponse,
    ResumeSessionResponse,
    SendTurnRequest,
    SendTurnResponse,
    SessionResponse,
    StartSessionRequest,
    StartSessionResponse,
)
from app.schemas.transcript_api import TranscriptResponse
from app.services.browse_service import list_sessions
from app.services.parser import ParseError
from app.services.session_resume_service import (
    SessionResumeError,
    get_session_for_resume,
)
from app.services.session_service import (
    SessionServiceError,
    abandon_session,
    approve_session,
    request_next_question,
    send_user_answer,
    start_session,
)
from app.services.transcript_service import (
    TranscriptServiceError,
    get_transcript,
)
from app.transport.base import LLMTransport, TransportError

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _map_service_error(exc: SessionServiceError) -> HTTPException:
    """Translate a service-layer error to an HTTP exception.

    Inspects message and cause to pick the right status code:
    not-found is 404, wrong-state is 409, transport or parse
    failures are 502 (the upstream LLM produced something wrong),
    and anything else is 500.

    The wrong-state checks substring-match on a known set of
    phrases. This is brittle (see the kind-discriminator
    pattern used by DiagnosticServiceError) and tracked as
    deferred work. New state-conflict phrases must be added here
    until the refactor lands.
    """
    message = exc.message
    if "not found" in message:
        return HTTPException(status.HTTP_404_NOT_FOUND, detail=message)
    if "expected in_progress" in message or "not awaiting" in message:
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


@router.get("", response_model=BrowseResponse)
async def browse(
    db: DbSession,
    state: SessionState | None = None,
) -> BrowseResponse:
    """List sessions sorted by created_at desc, optionally filtered by state.

    Hard limit of 50 rows. limit_reached on the response signals
    whether more sessions exist past the cap. The frontend can show
    a "more sessions exist" hint when True.
    """
    return list_sessions(db=db, state=state)


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
    """Send a user answer, return the grading response.

    After the split-roundtrip flow, the parsed reply is normally
    a ParsedGrading. The client signals Continue past the grading
    to request the next teaching turn via the continue route.
    """
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


@router.post("/{session_id}/continue", response_model=ContinueSessionResponse)
async def continue_session(
    session_id: str,
    db: DbSession,
    playwright: PlaywrightTransportDep,
    deepseek: DeepseekTransportDep,
) -> ContinueSessionResponse:
    """Request the next teaching turn after grading.

    Called after the client has displayed a grading response and
    the user (or the frontend prefetch) signals Continue. No body:
    the continue prompt is service-generated. Returns the next
    teaching turn, or rarely a session-end or handover.
    """
    session = db.get(Session, session_id)
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Session {session_id!r} not found.")

    transport = _pick_transport(session.transport_kind, playwright, deepseek)

    try:
        parsed = await request_next_question(
            db=db,
            transport=transport,
            session_id=session_id,
        )
    except SessionServiceError as exc:
        raise _map_service_error(exc) from exc

    return ContinueSessionResponse(parsed=parsed)


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


def _map_transcript_error(exc: TranscriptServiceError) -> HTTPException:
    """Translate a transcript-service error to an HTTP exception.

    not_found is 404, not_eligible is 409, malformed_parsed is
    500 (data integrity).
    """
    if exc.kind == "not_found":
        return HTTPException(status.HTTP_404_NOT_FOUND, detail=str(exc))
    if exc.kind == "not_eligible":
        return HTTPException(status.HTTP_409_CONFLICT, detail=str(exc))
    return HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.get("/{session_id}/transcript", response_model=TranscriptResponse)
async def transcript(
    session_id: str,
    db: DbSession,
) -> TranscriptResponse:
    """Return the user-visible transcript of a finished session.

    Available for COMPLETED, ABANDONED, and ARCHIVED sessions.
    IN_PROGRESS sessions return 409: the live session page is the
    right surface for an active session.
    """
    try:
        return get_transcript(db=db, session_id=session_id)
    except TranscriptServiceError as exc:
        raise _map_transcript_error(exc) from exc
