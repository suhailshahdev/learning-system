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

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from app.models import (
    LearnedItem,
    LearnedItemStatus,
    Session,
    SessionState,
    SessionTurn,
    Topic,
    TopicStatus,
    TransportKind,
    TurnRole,
)
from app.prompts.first_prompt import build_first_prompt
from app.prompts.handover_prompt import build_handover_request
from app.prompts.intro import build_intro
from app.prompts.turn_prompt import build_turn_prompt
from app.schemas.parsed_response import ParsedHandover, ParsedResponse, ParsedTurn
from app.services.parser import parse_response
from app.transport.base import (
    ChatResumeMetadata,
    PriorMessage,
    PriorRole,
    TransportError,
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
    topic = _get_or_create_topic(db, topic_path)

    intro = build_intro()
    first_prompt = build_first_prompt(topic_path)

    try:
        chat, response = await transport.start_new_chat(intro, first_prompt)
    except TransportError as e:
        db.rollback()
        raise SessionServiceError(
            f"Transport failed during session start: {e.message}", cause=e
        ) from e

    try:
        parsed = parse_response(response.text)
    except Exception as e:
        db.rollback()
        raise SessionServiceError("Parse failed on first response.", cause=e) from e

    if not isinstance(parsed, ParsedTurn):
        db.rollback()
        raise SessionServiceError(
            f"Expected a teaching turn on session start, got {parsed.kind!r}.",
        )

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
        db.rollback()
        raise SessionServiceError(
            f"Transport failed during send_user_answer: {e.message}", cause=e
        ) from e

    try:
        parsed = parse_response(response.text)
    except Exception as e:
        db.rollback()
        raise SessionServiceError("Parse failed on user-answer response.", cause=e) from e

    user_turn = SessionTurn(
        session_id=session.id,
        turn_index=next_index,
        role=TurnRole.USER,
        raw_content=answer,
        parsed=None,
        mode=None,
    )
    assistant_turn = SessionTurn(
        session_id=session.id,
        turn_index=next_index + 1,
        role=TurnRole.ASSISTANT,
        raw_content=response.text,
        parsed=parsed.model_dump(mode="json"),
        mode=parsed.mode if isinstance(parsed, ParsedTurn) else None,
    )
    db.add(user_turn)
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
        db.rollback()
        raise SessionServiceError(
            f"Transport failed during handover request: {e.message}", cause=e
        ) from e

    try:
        parsed = parse_response(handover_response.text)
    except Exception as e:
        db.rollback()
        raise SessionServiceError("Parse failed on handover response.", cause=e) from e

    if not isinstance(parsed, ParsedHandover):
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
    combined_intro = f"{build_intro()}\n\n---\n\n{_render_handover_block(handover)}"
    first_message = build_turn_prompt(answer)

    try:
        new_chat, new_response = await transport.start_new_chat(combined_intro, first_message)
    except TransportError as e:
        db.rollback()
        raise SessionServiceError(
            f"Transport failed opening new chat after handover: {e.message}", cause=e
        ) from e

    try:
        parsed = parse_response(new_response.text)
    except Exception as e:
        db.rollback()
        raise SessionServiceError(
            "Parse failed on new chat's first response after handover.", cause=e
        ) from e

    if not isinstance(parsed, ParsedTurn):
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


def _get_or_create_topic(db: DbSession, path: str) -> Topic:
    """Find a topic by path and create one if missing.

    Domain is denormalized from the first path segment per the
    Topic model's documented invariant.
    """
    existing = db.query(Topic).filter(Topic.path == path).one_or_none()
    if existing is not None:
        return existing

    domain = path.split(" > ", 1)[0]
    name = path.rsplit(" > ", 1)[-1]

    topic = Topic(
        path=path,
        domain=domain,
        name=name,
        status=TopicStatus.IN_PROGRESS,
    )
    db.add(topic)
    db.flush()
    return topic


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
    topic = _get_or_create_topic(db, parsed.topic_path)

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
