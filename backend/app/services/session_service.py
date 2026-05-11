"""Session service.

Orchestrates the lifecycle of a learning session: opening a chat
on a transport, sending prompts, parsing responses, persisting
turns, and minting learned items on approval. The service is the
only layer that knows about both the transport and the database;
transports do not write to the DB and DB models do not call
transports.

Covers session start, follow-up turns within the same chat, and
session approval. Auto-new-chat with handover and the abandoned-
state path are deferred to later steps.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from app.models import (
    ErrorLog,
    LearnedItem,
    LearnedItemStatus,
    Session,
    SessionState,
    SessionTurn,
    Topic,
    TransportKind,
    TurnRole,
)
from app.prompts.first_prompt import build_first_prompt
from app.prompts.handover_prompt import build_handover_request
from app.prompts.intro import build_intro
from app.prompts.turn_prompt import build_turn_prompt
from app.schemas.parsed_response import (
    ParsedHandover,
    ParsedResponse,
    ParsedToolCall,
    ParsedTurn,
)
from app.services.knowledge_service import derive_assertions_for_session
from app.services.parser import parse_response
from app.services.prereq_service import PrereqsUnmetError, check_prerequisites
from app.services.tools.handlers import ToolHandlerError
from app.services.tools.registry import execute_tool_call
from app.services.topic_crud import get_or_create_topic
from app.transport.base import (
    ChatResumeMetadata,
    PriorMessage,
    PriorRole,
    ToolResult,
    TransportError,
    TransportResponse,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession

    from app.transport.base import LLMTransport


class SessionServiceError(Exception):
    """A session-service operation failed.

    Wraps the underlying cause so callers see one error type at the
    service boundary. Specific failure modes (parse, transport,
    wrong response shape) are distinguishable via the cause chain
    when needed.
    """

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause


def _log_service_error(
    db: DbSession,
    *,
    kind: str,
    message: str,
    session_id: str | None,
    context: dict[str, Any] | None = None,
) -> None:
    """Write one row to error_log and commit it.

    Called from session-service catch blocks before the rollback and
    re-raise. Commits on the same database session to keep test
    isolation working.

    Relies on one invariant: session-service functions do not write
    any rows before the transport, parse, and validate block
    completes. Every catch site fires while the session is clean, so
    committing the log before rolling back is safe. If a future
    function breaks this invariant, the log commit will also persist
    any pending writes and break the all-or-nothing guarantee. Document
    it there if that ever happens.

    Errors inside this helper are swallowed so they do not mask the
    original error being raised.
    """
    try:
        row = ErrorLog(
            session_id=session_id,
            kind=kind,
            message=message,
            context=context or {},
        )
        db.add(row)
        db.commit()
    except Exception:
        # The original error is more important than the failure to log it.
        # Roll back the broken log write so the outer rollback does not
        # see a poisoned session.
        db.rollback()


async def start_session(
    *,
    db: DbSession,
    transport: LLMTransport[Any],
    transport_kind: TransportKind,
    topic_path: str,
) -> tuple[Session, ParsedTurn]:
    """Start a fresh session on the given topic.

    Resolves or creates the Topic, opens a chat on the transport,
    sends the first prompt, parses the response, and persists the
    Session plus the system and assistant turns. Commits on
    success and rolls back on any failure.

    transport_kind names which transport opened the chat. Stored
    on the session row so follow-up turns can route to the right
    resume_chat without inspecting the transport instance.

    Returns the persisted Session and the parsed first turn.
    """
    topic = get_or_create_topic(db, topic_path)

    unmet = check_prerequisites(db, topic_path)
    if unmet:
        raise PrereqsUnmetError(unmet)

    intro = await build_intro(db)
    first_prompt = build_first_prompt(topic_path)

    try:
        chat, response = await transport.start_new_chat(intro, first_prompt)
    except TransportError as e:
        _log_service_error(
            db,
            kind="session.start.transport_failed",
            message=e.message,
            session_id=None,
            context={"transport_kind": transport_kind.value, "topic_path": topic_path},
        )
        db.rollback()
        raise SessionServiceError(
            f"Transport failed during session start: {e.message}", cause=e
        ) from e

    try:
        parsed = _response_to_parsed(response)
    except Exception as e:
        _log_service_error(
            db,
            kind="session.start.parse_failed",
            message=str(e),
            session_id=None,
            context={
                "transport_kind": transport_kind.value,
                "topic_path": topic_path,
                "raw_response": response.text,
            },
        )
        db.rollback()
        raise SessionServiceError("Parse failed on first response.", cause=e) from e

    response, parsed, _ = await _execute_until_terminal(
        db=db,
        transport=transport,
        chat=chat,
        response=response,
        parsed=parsed,
        session_id=None,
        transport_kind=transport_kind,
        base_turn_index=0,
    )

    if not isinstance(parsed, ParsedTurn):
        _log_service_error(
            db,
            kind="session.start.wrong_response_kind",
            message=f"Expected ParsedTurn, got {parsed.kind!r}.",
            session_id=None,
            context={
                "transport_kind": transport_kind.value,
                "topic_path": topic_path,
                "actual_kind": parsed.kind,
                "expected_kind": "turn",
            },
        )
        db.rollback()
        raise SessionServiceError(
            f"Expected a teaching turn on session start, got {parsed.kind!r}.",
        )

    _populate_topic_prerequisites(topic, parsed)

    session = _build_session(
        topic=topic,
        parsed=parsed,
        chat=chat,
        transport_kind=transport_kind,
    )
    db.add(session)
    db.flush()  # populates session.id for FK on the turns

    db.add(_build_system_turn(session_id=session.id, intro=intro, first_prompt=first_prompt))
    db.add(_build_assistant_turn(session_id=session.id, response_text=response.text, parsed=parsed))

    db.commit()
    db.refresh(session)
    return session, parsed


async def send_user_answer(
    *,
    db: DbSession,
    transport: LLMTransport[Any],
    session_id: str,
    answer: str,
) -> ParsedResponse:
    """Send the user's answer in an in-progress session and parse the reply.

    Resumes the LLM chat from persisted session state, sends the
    user's answer, parses the response, and persists the new turns
    in one transaction. Returns the parsed response so the caller can
    branch on its kind: a ParsedTurn means continue the session, a
    ParsedSessionEnd means the LLM proposes wrapping up, and a
    ParsedHandover means the chat itself emitted a handover block.

    When the session's chat is at or past HANDOVER_THRESHOLD, this
    call splits the chat: the dying chat produces a handover, a
    fresh chat opens with that handover seeded into its intro, and
    the user's answer goes through the new chat. The session row
    stays the same and turns persist in unbroken turn_index order.

    The session must be in IN_PROGRESS state. Either all new turns
    are written or none are.
    """
    session = db.get(Session, session_id)
    if session is None:
        raise SessionServiceError(f"Session {session_id!r} not found.")
    if session.state is not SessionState.IN_PROGRESS:
        raise SessionServiceError(
            f"Session {session_id!r} is in state {session.state.value!r}, expected in_progress.",
        )

    if session.claude_chat_message_count >= HANDOVER_THRESHOLD:
        return await _send_with_handover(db=db, transport=transport, session=session, answer=answer)
    return await _send_within_chat(db=db, transport=transport, session=session, answer=answer)


async def _send_within_chat(
    *,
    db: DbSession,
    transport: LLMTransport[Any],
    session: Session,
    answer: str,
) -> ParsedResponse:
    """Send the user's answer inside the existing chat.

    The default path: chat has budget remaining, resume it, send
    one turn, parse, persist user + assistant turns.
    """
    metadata = _rebuild_chat_metadata(session)
    next_index = _next_turn_index(db, session.id)
    prompt = build_turn_prompt(answer)

    try:
        chat = await transport.resume_chat(metadata)
        response = await transport.send(chat, prompt)
    except TransportError as e:
        _log_service_error(
            db,
            kind="session.send.transport_failed",
            message=e.message,
            session_id=session.id,
            context={"transport_kind": session.transport_kind.value},
        )
        db.rollback()
        raise SessionServiceError(
            f"Transport failed during send_user_answer: {e.message}", cause=e
        ) from e

    try:
        parsed = _response_to_parsed(response)
    except Exception as e:
        _log_service_error(
            db,
            kind="session.send.parse_failed",
            message=str(e),
            session_id=session.id,
            context={
                "transport_kind": session.transport_kind.value,
                "raw_response": response.text,
            },
        )
        db.rollback()
        raise SessionServiceError("Parse failed on user-answer response.", cause=e) from e

    # The user's answer goes in first so its turn_index is before any
    # tool turns the helper persists. Helper indexes start at next_index + 1.
    user_turn = SessionTurn(
        session_id=session.id,
        turn_index=next_index,
        role=TurnRole.USER,
        raw_content=answer,
        parsed=None,
        mode=None,
    )
    db.add(user_turn)
    db.flush()

    response, parsed, after_tools_index = await _execute_until_terminal(
        db=db,
        transport=transport,
        chat=chat,
        response=response,
        parsed=parsed,
        session_id=session.id,
        transport_kind=session.transport_kind,
        base_turn_index=next_index + 1,
    )

    assistant_turn = SessionTurn(
        session_id=session.id,
        turn_index=after_tools_index,
        role=TurnRole.ASSISTANT,
        raw_content=response.text,
        parsed=parsed.model_dump(mode="json"),
        mode=parsed.mode if isinstance(parsed, ParsedTurn) else None,
    )
    db.add(assistant_turn)

    session.claude_chat_message_count = getattr(chat, "message_count", 0)
    if isinstance(parsed, ParsedTurn):
        session.mode_used = parsed.mode

    db.commit()
    db.refresh(session)
    return parsed


async def _send_with_handover(
    *,
    db: DbSession,
    transport: LLMTransport[Any],
    session: Session,
    answer: str,
) -> ParsedResponse:
    """Split the chat: handover the dying chat, open a new one, deliver the answer there.

    Five turns persist on success: SYSTEM (handover request prompt),
    ASSISTANT (handover response from dying chat), TRANSITION (the
    standard handover block carried over), USER (the user's answer
    in the new chat), ASSISTANT (the LLM's response in the new chat).

    Any failure rolls back the entire transition. The caller sees a
    SessionServiceError and the session row and prior turns are
    untouched.
    """
    handover_block = await _request_and_parse_handover(db=db, transport=transport, session=session)
    new_chat, new_response, new_parsed = await _open_new_chat_with_handover(
        db=db, transport=transport, session=session, handover=handover_block, answer=answer
    )

    next_index = _next_turn_index(db, session.id)
    handover_request_text = build_handover_request()

    db.add(
        SessionTurn(
            session_id=session.id,
            turn_index=next_index,
            role=TurnRole.SYSTEM,
            raw_content=handover_request_text,
            parsed=None,
            mode=None,
        )
    )
    db.add(
        SessionTurn(
            session_id=session.id,
            turn_index=next_index + 1,
            role=TurnRole.ASSISTANT,
            raw_content=_handover_response_marker(handover_block),
            parsed=handover_block.model_dump(mode="json"),
            mode=None,
        )
    )
    db.add(
        SessionTurn(
            session_id=session.id,
            turn_index=next_index + 2,
            role=TurnRole.TRANSITION,
            raw_content=_render_handover_block(handover_block),
            parsed=handover_block.model_dump(mode="json"),
            mode=None,
        )
    )
    db.add(
        SessionTurn(
            session_id=session.id,
            turn_index=next_index + 3,
            role=TurnRole.USER,
            raw_content=answer,
            parsed=None,
            mode=None,
        )
    )
    db.add(
        SessionTurn(
            session_id=session.id,
            turn_index=next_index + 4,
            role=TurnRole.ASSISTANT,
            raw_content=new_response.text,
            parsed=new_parsed.model_dump(mode="json"),
            mode=new_parsed.mode,
        )
    )

    session.claude_chat_url = getattr(new_chat, "chat_url", None)
    session.claude_chat_message_count = getattr(new_chat, "message_count", 0)
    session.mode_used = new_parsed.mode

    db.commit()
    db.refresh(session)
    return new_parsed


async def _request_and_parse_handover(
    *,
    db: DbSession,
    transport: LLMTransport[Any],
    session: Session,
) -> ParsedHandover:
    """Resume the dying chat, request a handover, parse and validate the response.

    The dying chat's response must be a ParsedHandover. Anything
    else is treated as a transition failure and rolls back.
    """
    metadata = _rebuild_chat_metadata(session)

    try:
        old_chat = await transport.resume_chat(metadata)
        handover_response = await transport.send(old_chat, build_handover_request())
        await transport.close(old_chat)
    except TransportError as e:
        _log_service_error(
            db,
            kind="session.handover.request_transport_failed",
            message=e.message,
            session_id=session.id,
            context={"transport_kind": session.transport_kind.value},
        )
        db.rollback()
        raise SessionServiceError(
            f"Transport failed during handover request: {e.message}", cause=e
        ) from e

    try:
        parsed = _response_to_parsed(handover_response)
    except Exception as e:
        _log_service_error(
            db,
            kind="session.handover.request_parse_failed",
            message=str(e),
            session_id=session.id,
            context={
                "transport_kind": session.transport_kind.value,
                "raw_response": handover_response.text,
            },
        )
        db.rollback()
        raise SessionServiceError("Parse failed on handover response.", cause=e) from e

    # A handover-request response should land directly. Tool calls in
    # this narrow path would chain extra turns onto a dying chat and
    # bloat the next chat's intro. Reject defensively.
    if isinstance(parsed, ParsedToolCall):
        _log_service_error(
            db,
            kind="session.handover.unexpected_tool_call",
            message=f"Tool call {parsed.call.name!r} in handover request response.",
            session_id=session.id,
            context={
                "transport_kind": session.transport_kind.value,
                "tool_name": parsed.call.name,
            },
        )
        db.rollback()
        raise SessionServiceError(
            f"Unexpected tool call in handover request: {parsed.call.name!r}.",
        )

    if not isinstance(parsed, ParsedHandover):
        _log_service_error(
            db,
            kind="session.handover.wrong_response_kind",
            message=f"Expected ParsedHandover, got {parsed.kind!r}.",
            session_id=session.id,
            context={
                "transport_kind": session.transport_kind.value,
                "actual_kind": parsed.kind,
                "expected_kind": "handover",
            },
        )
        db.rollback()
        raise SessionServiceError(
            f"Expected a handover block from dying chat, got {parsed.kind!r}.",
        )
    return parsed


async def _open_new_chat_with_handover(
    *,
    db: DbSession,
    transport: LLMTransport[Any],
    session: Session,
    handover: ParsedHandover,
    answer: str,
) -> tuple[Any, Any, ParsedTurn]:
    """Open a fresh chat with the handover seeded into its intro.

    The new chat sees the original intro, the handover block, and
    the user's answer as its first message. The response must be a
    ParsedTurn; the new chat shouldn't propose session end on its
    very first reply.

    Returns the new chat handle, the raw response, and the parsed
    teaching turn.
    """
    combined_intro = f"{await build_intro(db)}\n\n---\n\n{_render_handover_block(handover)}"
    first_message = build_turn_prompt(answer)

    try:
        new_chat, new_response = await transport.start_new_chat(combined_intro, first_message)
    except TransportError as e:
        _log_service_error(
            db,
            kind="session.handover.new_chat_transport_failed",
            message=e.message,
            session_id=session.id,
            context={"transport_kind": session.transport_kind.value},
        )
        db.rollback()
        raise SessionServiceError(
            f"Transport failed opening new chat after handover: {e.message}", cause=e
        ) from e

    try:
        parsed = _response_to_parsed(new_response)
    except Exception as e:
        _log_service_error(
            db,
            kind="session.handover.new_chat_parse_failed",
            message=str(e),
            session_id=session.id,
            context={
                "transport_kind": session.transport_kind.value,
                "raw_response": new_response.text,
            },
        )
        db.rollback()
        raise SessionServiceError(
            "Parse failed on new chat's first response after handover.", cause=e
        ) from e

    # Tool calls on a new chat's very first reply land between
    # TRANSITION and the new chat's first USER turn in the natural
    # flow, but _send_with_handover persists all 5 transition turns
    # in one block after this function returns. Mixing helper-written
    # tool turns into that ordering needs a real refactor. Reject
    # defensively until real data shows this path matters.
    if isinstance(parsed, ParsedToolCall):
        _log_service_error(
            db,
            kind="session.handover.new_chat_unexpected_tool_call",
            message=f"Tool call {parsed.call.name!r} on new chat's first response.",
            session_id=session.id,
            context={
                "transport_kind": session.transport_kind.value,
                "tool_name": parsed.call.name,
            },
        )
        db.rollback()
        raise SessionServiceError(
            f"Unexpected tool call after handover: {parsed.call.name!r}.",
        )

    if not isinstance(parsed, ParsedTurn):
        _log_service_error(
            db,
            kind="session.handover.new_chat_wrong_response_kind",
            message=f"Expected ParsedTurn, got {parsed.kind!r}.",
            session_id=session.id,
            context={
                "transport_kind": session.transport_kind.value,
                "actual_kind": parsed.kind,
                "expected_kind": "turn",
            },
        )
        db.rollback()
        raise SessionServiceError(
            f"Expected a teaching turn after handover, got {parsed.kind!r}.",
        )
    return new_chat, new_response, parsed


def _render_handover_block(handover: ParsedHandover) -> str:
    """Reconstruct the standard wire format from a ParsedHandover.

    The dying chat's response may have had conversational intro
    that the parser tolerated. Reconstructing from the structured
    fields gives the new chat (and any future replay) a clean
    standard shape regardless of what the dying chat actually
    produced.
    """
    return (
        "---HANDOVER---\n"
        f"DOMAIN_FOCUS: {handover.domain_focus}\n"
        f"COVERED: {handover.covered}\n"
        f"LAST_QUESTION: {handover.last_question}\n"
        f"NEXT_PLANNED: {handover.next_planned}\n"
        f"OPEN_THREADS: {handover.open_threads}\n"
        f"USER_STATE: {handover.user_state}\n"
        "---END_HANDOVER---"
    )


def _handover_response_marker(handover: ParsedHandover) -> str:
    """Marker text stored in the assistant turn's raw_content for the handover response.

    The actual structured handover lives in the turn's parsed JSON.
    raw_content gets a short human-readable summary so admin CLI
    output and grep stay useful without dumping the whole block.
    """
    return f"[handover requested by service; structured fields in parsed]\n{handover.last_question}"


def _response_to_parsed(response: TransportResponse) -> ParsedResponse:
    """Translate a TransportResponse into a ParsedResponse.

    Two paths produce tool calls. Claude transport emits them as
    ---TOOL_CALL--- blocks in chat text, which parse_response
    yields as ParsedToolCall. DeepSeek transport surfaces them
    via the API's native function calling, populating
    TransportResponse.tool_calls directly with the response text
    empty. Both must converge on the same ParsedToolCall shape so
    the session-service loop handles them uniformly.

    When tool_calls is non-empty, synthesize a ParsedToolCall for
    the first entry. The current wire format and helper loop process
    one tool call per iteration. If the API returned multiple, only
    the first is handled per pass and the helper loops to handle
    the rest. raw_text is reconstructed from the call's JSON shape
    so error_log entries stay greppable in the same shape as the
    Claude path.

    Otherwise fall through to text parsing.
    """
    if response.tool_calls:
        call = response.tool_calls[0]
        raw_text = call.model_dump_json()
        return ParsedToolCall(call=call, raw_text=raw_text)
    return parse_response(response.text)


async def _execute_until_terminal(
    *,
    db: DbSession,
    transport: LLMTransport[Any],
    chat: Any,
    response: TransportResponse,
    parsed: ParsedResponse,
    session_id: str | None,
    transport_kind: TransportKind,
    base_turn_index: int,
) -> tuple[TransportResponse, ParsedResponse, int]:
    """Drive the tool-execution loop until a terminal response lands.

    Called from every transport-call site (start_session, _send_within_chat,
    handover paths). Takes the transport's response and the parsed
    shape. If parsed is a ParsedToolCall, executes the handler,
    persists a TOOL_CALL + TOOL_RESULT turn pair, sends the result
    back to the transport, parses the next response, and loops.

    The ParsedToolCall variant never escapes this helper: by return
    time, parsed is guaranteed to be ParsedTurn, ParsedSessionEnd, or
    ParsedHandover. Callers branch on those three kinds without
    worrying about tool calls.

    session_id is None when called from start_session before the
    Session row exists. Tool-call turns are not persisted in that
    case (the FK would fail) but the registry calls still run. This
    is the only path that allows tool execution without persistence.
    In practice, tool calls during session start are rare and the
    handler's own commit captures any state change.

    Returns the final (response, parsed, next_turn_index) so callers
    can persist their own turns starting at the returned index.
    """
    next_index = base_turn_index

    while isinstance(parsed, ParsedToolCall):
        # Execute the handler. The registry commits handler writes
        # independently so a later failure to produce a valid
        # teaching turn does not roll back correct system-state changes.
        try:
            output = await execute_tool_call(db, parsed.call)
            content = output.model_dump_json()
        except ToolHandlerError as e:
            # Roll back first so pending writes from the caller (e.g. the
            # user turn added in _send_within_chat before this helper ran)
            # are discarded before the log commit.
            db.rollback()
            _log_service_error(
                db,
                kind="session.tool_call.handler_failed",
                message=e.message,
                session_id=session_id,
                context={
                    "transport_kind": transport_kind.value,
                    "tool_name": parsed.call.name,
                    "raw_text": parsed.raw_text,
                },
            )
            raise SessionServiceError(
                f"Tool handler {parsed.call.name!r} failed: {e.message}", cause=e
            ) from e

        # Persist the TOOL_CALL + TOOL_RESULT turn pair. Skipped when
        # session_id is None (start_session pre-session-row case).
        if session_id is not None:
            db.add(
                SessionTurn(
                    session_id=session_id,
                    turn_index=next_index,
                    role=TurnRole.TOOL_CALL,
                    raw_content=parsed.raw_text,
                    parsed=parsed.model_dump(mode="json"),
                    mode=None,
                )
            )
            db.add(
                SessionTurn(
                    session_id=session_id,
                    turn_index=next_index + 1,
                    role=TurnRole.TOOL_RESULT,
                    raw_content=content,
                    parsed=json.loads(content),
                    mode=None,
                )
            )
            db.flush()
            next_index += 2

        # Send the result back to the transport. The LLM may respond
        # with another tool call (chained execution) or with terminal
        # content. The loop continues either way.
        call_id = _tool_call_id(parsed)
        result = ToolResult(call_id=call_id, content=content)

        try:
            response = await transport.send_tool_results(chat, [result])
        except TransportError as e:
            db.rollback()
            _log_service_error(
                db,
                kind="session.tool_call.send_results_failed",
                message=e.message,
                session_id=session_id,
                context={
                    "transport_kind": transport_kind.value,
                    "tool_name": parsed.call.name,
                },
            )
            raise SessionServiceError(
                f"Transport failed sending tool results: {e.message}", cause=e
            ) from e

        try:
            parsed = _response_to_parsed(response)
        except Exception as e:
            db.rollback()
            _log_service_error(
                db,
                kind="session.tool_call.parse_failed",
                message=str(e),
                session_id=session_id,
                context={
                    "transport_kind": transport_kind.value,
                    "raw_response": response.text,
                },
            )
            raise SessionServiceError("Parse failed on response after tool result.", cause=e) from e

    return response, parsed, next_index


def _tool_call_id(parsed: ParsedToolCall) -> str:
    """Extract the call id from a ParsedToolCall.

    DeepSeek's native function calling requires a call_id for
    correlating results back to the originating call: the API
    rejects subsequent requests if the `tool` role message's
    tool_call_id does not match the assistant message's
    tool_calls[i].id. The Claude transport has no id concept in
    its chat wire format, so its ToolCall.id is None and the
    tool name is the stable fallback.
    """
    return parsed.call.id or parsed.call.name


def _next_turn_index(db: DbSession, session_id: str) -> int:
    """Return the next turn_index for the given session."""
    last = (
        db.query(SessionTurn)
        .filter(SessionTurn.session_id == session_id)
        .order_by(SessionTurn.turn_index.desc())
        .first()
    )
    return 0 if last is None else last.turn_index + 1


# Maps DB-side TurnRole values to transport-side PriorRole literals.
# TRANSITION turns are persistence-only markers (they record where a
# chat handover happened) and do not belong in replay history.
_PRIOR_ROLE_BY_TURN_ROLE: dict[TurnRole, PriorRole] = {
    TurnRole.SYSTEM: "system",
    TurnRole.USER: "user",
    TurnRole.ASSISTANT: "assistant",
}


def _rebuild_chat_metadata(session: Session) -> ChatResumeMetadata:
    """Build ChatResumeMetadata from a persisted session and its turns.

    chat_url comes straight from the session row. prior_messages is
    rebuilt from session_turn rows in turn order, skipping turns that
    do not represent real conversation messages (TRANSITION). Transports
    that only need chat_url (Playwright) ignore prior_messages entirely.
    """
    prior_messages: list[PriorMessage] = []
    for turn in session.turns:
        prior_role = _PRIOR_ROLE_BY_TURN_ROLE.get(turn.role)
        if prior_role is None:
            continue
        prior_messages.append(
            PriorMessage(
                role=prior_role,
                content=turn.raw_content,
            )
        )
    return ChatResumeMetadata(
        chat_url=session.claude_chat_url,
        prior_messages=prior_messages,
        message_count=session.claude_chat_message_count,
    )


def _populate_topic_prerequisites(topic: Topic, parsed: ParsedTurn) -> None:
    """Write the first response's prereqs onto the topic if not already set.

    Topics start with empty prerequisites. The first parsed turn for
    a topic fills them in from the LLM's response. After that the
    column is left alone since later sessions should not silently
    overwrite the original list. Manual edits via the topic editor
    are the right place to revise them.
    """
    if topic.prerequisites:
        return
    topic.prerequisites = [p.model_dump(mode="json") for p in parsed.prerequisites]


def _build_session(
    *,
    topic: Topic,
    parsed: ParsedTurn,
    chat: Any,
    transport_kind: TransportKind,
) -> Session:
    """Construct an in-memory Session for the new session start."""
    return Session(
        topic_id=topic.id,
        mode_used=parsed.mode,
        state=SessionState.IN_PROGRESS,
        transport_kind=transport_kind,
        claude_chat_url=getattr(chat, "chat_url", None),
        claude_chat_message_count=getattr(chat, "message_count", 0),
        active_preferences=[],
        context_snapshot={},
    )


def _build_system_turn(*, session_id: str, intro: str, first_prompt: str) -> SessionTurn:
    """Build the system-role turn capturing the intro plus the kickoff prompt."""
    return SessionTurn(
        session_id=session_id,
        turn_index=0,
        role=TurnRole.SYSTEM,
        raw_content=f"{intro}\n\n---\n\n{first_prompt}",
        parsed=None,
        mode=None,
    )


def _build_assistant_turn(
    *, session_id: str, response_text: str, parsed: ParsedTurn
) -> SessionTurn:
    """Build the assistant-role turn from the LLM's first response."""
    return SessionTurn(
        session_id=session_id,
        turn_index=1,
        role=TurnRole.ASSISTANT,
        raw_content=response_text,
        parsed=parsed.model_dump(mode="json"),
        mode=parsed.mode,
    )


# Placeholder stored in learned_item.answer when the LLM graded the
# turn conversationally (EXPECTED_ANSWER was OPEN). The column is
# non-nullable; this preserves the item with a clear marker rather
# than dropping it or storing an empty string.
OPEN_ANSWER_PLACEHOLDER = "[graded conversationally]"


# Maximum user-turn count per LLM chat before the next send_user_answer
# call triggers a chat transition. Conservative for claude.ai's free-plan
# limit. DeepSeek has no real cap but a long history bloats every request
# payload. Tuned empirically as real session data accumulates.
HANDOVER_THRESHOLD = 30


async def approve_session(*, db: DbSession, session_id: str) -> Session:
    """Approve an in-progress session and mint learned items.

    Walks the session's turns in order, pairs each parseable
    teaching turn with the user's next answer, and writes one
    LearnedItem per pair. Marks the session COMPLETED. All writes
    commit together or roll back together.

    Returns the refreshed Session.
    """
    session = db.get(Session, session_id)
    if session is None:
        raise SessionServiceError(f"Session {session_id!r} not found.")
    if session.state is not SessionState.IN_PROGRESS:
        raise SessionServiceError(
            f"Session {session_id!r} is in state {session.state.value!r}, expected in_progress.",
        )

    now = datetime.now(UTC)
    items = _build_learned_items(db, session, now)

    for item in items:
        db.add(item)

    # Flush so derivation's queries see the items we just minted alongside
    # any historical items for the same (topic, difficulty) pair.
    db.flush()

    derive_assertions_for_session(db, session)
    session.state = SessionState.COMPLETED

    db.commit()
    db.refresh(session)
    return session


def _build_learned_items(db: DbSession, session: Session, now: datetime) -> list[LearnedItem]:
    """Build one LearnedItem per teaching turn that has a user answer.

    Pairs each ASSISTANT turn whose parsed payload is a teaching
    turn with the immediately following USER turn. Teaching turns
    without a user answer (e.g. an unanswered final question
    before SESSION_END_PROPOSAL) are skipped.
    """
    turns = sorted(session.turns, key=lambda t: t.turn_index)
    items: list[LearnedItem] = []

    for i, turn in enumerate(turns):
        if turn.role is not TurnRole.ASSISTANT or turn.parsed is None:
            continue
        if turn.parsed.get("kind") != "turn":
            continue

        next_turn = turns[i + 1] if i + 1 < len(turns) else None
        if next_turn is None or next_turn.role is not TurnRole.USER:
            continue

        items.append(_build_learned_item(db, turn, next_turn, now))

    return items


def _build_learned_item(
    db: DbSession,
    assistant_turn: SessionTurn,
    user_turn: SessionTurn,
    now: datetime,
) -> LearnedItem:
    """Build one LearnedItem from a (ParsedTurn, user-answer) pair."""
    parsed = ParsedTurn.model_validate(assistant_turn.parsed)
    topic = get_or_create_topic(db, parsed.topic_path)

    answer = parsed.expected_answer or OPEN_ANSWER_PLACEHOLDER

    return LearnedItem(
        session_id=assistant_turn.session_id,
        topic_id=topic.id,
        question=parsed.question,
        answer=answer,
        your_answer=user_turn.raw_content,
        mode=parsed.mode,
        difficulty=parsed.difficulty,
        status=LearnedItemStatus.LEARNED,
        last_reviewed_at=now,
    )


async def abandon_session(*, db: DbSession, session_id: str) -> Session:
    """Abandon an in-progress session without minting learned items.

    The user closing the tab or hitting "End session" without
    approving lands here. No learned items are written: the partial
    Q/A pairs from this session stay only as session_turn rows for
    replay. Marks the session ABANDONED and commits.

    Returns the refreshed Session.
    """
    session = db.get(Session, session_id)
    if session is None:
        raise SessionServiceError(f"Session {session_id!r} not found.")
    if session.state is not SessionState.IN_PROGRESS:
        raise SessionServiceError(
            f"Session {session_id!r} is in state {session.state.value!r}, expected in_progress.",
        )

    session.state = SessionState.ABANDONED
    db.commit()
    db.refresh(session)
    return session
