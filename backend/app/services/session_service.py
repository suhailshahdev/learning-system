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
    TurnRole,
)
from app.prompts.first_prompt import build_first_prompt
from app.prompts.intro import build_intro
from app.prompts.turn_prompt import build_turn_prompt
from app.schemas.parsed_response import ParsedResponse, ParsedTurn
from app.services.parser import parse_response
from app.transport.base import (
    ChatResumeMetadata,
    PriorMessage,
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
    topic_path: str,
) -> tuple[Session, ParsedTurn]:
    """Start a fresh session on the given topic.

    Resolves or creates the Topic, opens a chat on the transport,
    sends the first prompt, parses the response, and persists the
    Session plus the system and assistant turns. Commits on
    success and rolls back on any failure.

    Returns the persisted Session and the parsed first turn.
    """
    topic = _get_or_create_topic(db, topic_path)

    intro = build_intro()
    first_prompt = build_first_prompt(topic_path)

    try:
        chat = await transport.start_new_chat(intro)
        response = await transport.send(chat, first_prompt)
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

    session = _build_session(topic=topic, parsed=parsed, chat=chat)
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
    user's answer, parses the response, and persists both turns in
    one transaction. Returns the parsed response so the caller can
    branch on its kind: a ParsedTurn means continue the session, a
    ParsedSessionEnd means the LLM proposes wrapping up, and a
    ParsedHandover means the chat is near its message count threshold.

    The session must be in IN_PROGRESS state. Both turns are
    written or neither is.
    """
    session = db.get(Session, session_id)
    if session is None:
        raise SessionServiceError(f"Session {session_id!r} not found.")
    if session.state is not SessionState.IN_PROGRESS:
        raise SessionServiceError(
            f"Session {session_id!r} is in state {session.state.value!r}, expected in_progress.",
        )

    metadata = _rebuild_chat_metadata(session)
    next_index = _next_turn_index(db, session_id)
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


def _next_turn_index(db: DbSession, session_id: str) -> int:
    """Return the next turn_index for the given session."""
    last = (
        db.query(SessionTurn)
        .filter(SessionTurn.session_id == session_id)
        .order_by(SessionTurn.turn_index.desc())
        .first()
    )
    return 0 if last is None else last.turn_index + 1


def _rebuild_chat_metadata(session: Session) -> ChatResumeMetadata:
    """Build ChatResumeMetadata from a persisted session and its turns.

    chat_url comes straight from the session row. prior_messages is
    rebuilt from session_turn rows in turn order, and transports that
    only need chat_url (Playwright) ignore it. Each turn's role is
    mapped to the transport-side PriorRole literal.
    """
    prior_messages: list[PriorMessage] = []
    for turn in session.turns:
        prior_messages.append(
            PriorMessage(
                role=turn.role.value,
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


def _build_session(*, topic: Topic, parsed: ParsedTurn, chat: Any) -> Session:
    """Construct an in-memory Session for the new session start."""
    return Session(
        topic_id=topic.id,
        mode_used=parsed.mode,
        state=SessionState.IN_PROGRESS,
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
