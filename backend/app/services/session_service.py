"""Session service.

Orchestrates the lifecycle of a learning session: opening a chat
on a transport, sending prompts, parsing responses, and persisting
turns. The service is the only layer that knows about both the
transport and the database; transports do not write to the DB and
DB models do not call transports.

This file currently covers session start (one round-trip from
intro through first parsed turn). Multi-turn flow, pause/resume,
auto-new-chat with handover, and end states are deferred to a
future step.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.models import (
    Session,
    SessionState,
    SessionTurn,
    Topic,
    TopicStatus,
    TurnRole,
)
from app.prompts.first_prompt import build_first_prompt
from app.prompts.intro import build_intro
from app.schemas.parsed_response import ParsedTurn
from app.services.parser import parse_response
from app.transport.base import TransportError

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
