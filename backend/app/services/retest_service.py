"""Retest service.

Creates a new session linked to a completed source session, walking
the same questions the source produced. The user gets to redo the
material and the source session row stays untouched.

Lazy transport: start_retest does not open an LLM chat. The retest
session's questions are reconstructed from the source's LearnedItems,
not LLM-generated. A chat opens later, only if and when a question
needs LLM grading (free-form modes like explain_back or socratic).
Deterministic-mode questions (flashcard, type_the_answer,
multiple_choice) never trigger a chat open.

The first question is materialized as a synthetic ASSISTANT turn
at turn_index 0 with a fully-reconstructed ParsedTurn payload. This
keeps the resume, transcript, and approve flows working without
retest-specific branches: the retest session looks like any other
session that happens to have started with a teaching turn already
in place.

Question ordering: source LearnedItems are walked by created_at
ascending, reproducing the original session's order. Random-order
and failed-first ordering modes are deferred to a follow-up.
"""

from __future__ import annotations

import contextlib
import json
from typing import TYPE_CHECKING

from sqlalchemy import select

from app.models import (
    Difficulty,
    LearnedItem,
    Session,
    SessionState,
    SessionTurn,
    Topic,
    TransportKind,
    TurnRole,
)
from app.prompts.retest_grading_intro import (
    build_retest_grading_intro,
    build_retest_grading_prompt,
)
from app.schemas.parsed_response import ParsedGrading, ParsedToolCall, ParsedTurn
from app.services.parser import parse_response
from app.services.session_service import OPEN_ANSWER_PLACEHOLDER
from app.transport.base import TransportError, TransportResponse

if TYPE_CHECKING:
    from typing import Any

    from sqlalchemy.orm import Session as DbSession

    from app.transport.base import LLMTransport


class RetestServiceError(Exception):
    """A retest-service operation failed.

    Carries a kind discriminator so the route layer maps to HTTP
    status codes without string-matching the message. Same pattern
    as DiagnosticServiceError and TranscriptServiceError.
    """

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


def start_retest(
    db: DbSession,
    *,
    source_session_id: str,
    transport_kind: TransportKind,
) -> tuple[Session, ParsedTurn]:
    """Start a retest of the named source session.

    Validates that the source session is COMPLETED and has at least
    one LearnedItem. Creates a new IN_PROGRESS session with
    parent_session_id pointing at the source. Materializes the
    source's first LearnedItem as a synthetic ASSISTANT turn at
    turn_index 0 so resume, transcript, and approve flows work
    unchanged.

    transport_kind is stored on the row for later use when grading
    fires. The transport itself is not contacted here.

    Raises RetestServiceError with kind:
    - "not_found" when source_session_id is unknown
    - "not_eligible" when source is not COMPLETED
    - "empty_source" when source has no learned items to retest
    """
    source = db.get(Session, source_session_id)
    if source is None:
        raise RetestServiceError("not_found", f"Source session {source_session_id} not found")
    if source.state is not SessionState.COMPLETED:
        raise RetestServiceError(
            "not_eligible",
            f"Source session {source_session_id} is in state {source.state.value}; "
            "only completed sessions can be retested",
        )

    source_items = (
        db.execute(
            select(LearnedItem)
            .where(LearnedItem.session_id == source_session_id)
            .order_by(LearnedItem.created_at.asc())
        )
        .scalars()
        .all()
    )
    if not source_items:
        raise RetestServiceError(
            "empty_source",
            f"Source session {source_session_id} has no learned items to retest",
        )

    first_item = source_items[0]
    first_turn = _reconstruct_parsed_turn(db, first_item)

    retest_session = Session(
        topic_id=source.topic_id,
        parent_session_id=source.id,
        mode_used=source.mode_used,
        state=SessionState.IN_PROGRESS,
        transport_kind=transport_kind,
        claude_chat_url=None,
        claude_chat_message_count=0,
        active_preferences=[],
        context_snapshot={},
    )
    db.add(retest_session)
    db.flush()  # populate retest_session.id for the synthetic turn's FK

    synthetic_turn = SessionTurn(
        session_id=retest_session.id,
        turn_index=0,
        role=TurnRole.ASSISTANT,
        raw_content=_synthetic_raw_content(first_item),
        parsed=first_turn.model_dump(mode="json"),
        mode=first_item.mode,
    )
    db.add(synthetic_turn)

    db.commit()
    db.refresh(retest_session)
    return retest_session, first_turn


def _reconstruct_parsed_turn(db: DbSession, item: LearnedItem) -> ParsedTurn:
    """Build a ParsedTurn from a source LearnedItem.

    LearnedItem only carries the inputs the user saw: question,
    canonical answer, mode, difficulty. The other ParsedTurn fields
    (prerequisites, requirements, followup, tags, question_code) are
    not preserved on LearnedItem, so they reconstruct as empty/None.

    topic_path comes from the LearnedItem's joined Topic row. The
    item carries topic_id which always points to a real Topic. The
    path lookup is one query per first item, acceptable cost.

    Difficulty is non-nullable on the wire format but nullable on
    LearnedItem. Falls back to BEGINNER when unset. This is purely
    a render-time concern (the retest user sees the question, not
    the difficulty pill).
    """
    topic = db.get(Topic, item.topic_id)
    if topic is None:
        # Shouldn't happen given FK with ondelete=RESTRICT on
        # LearnedItem.topic_id, but defend rather than assume.
        raise RetestServiceError(
            "not_found",
            f"Topic {item.topic_id} for learned item {item.id} not found",
        )

    return ParsedTurn(
        topic_path=topic.path,
        difficulty=item.difficulty or Difficulty.BEGINNER,
        prerequisites=[],
        mode=item.mode,
        question=item.question,
        question_code=None,
        expected_answer=_canonical_answer(item),
        requirements=None,
        followup=None,
        tags=[],
    )


def _canonical_answer(item: LearnedItem) -> str | None:
    """Return the canonical expected_answer for retest grading.

    Source items minted with EXPECTED_ANSWER=OPEN carry the
    OPEN_ANSWER_PLACEHOLDER string. Convert that back to None so the
    reconstructed ParsedTurn matches the original wire format: None
    in the field signals open-graded to downstream code.
    """
    if item.answer == OPEN_ANSWER_PLACEHOLDER:
        return None
    return item.answer


def _synthetic_raw_content(item: LearnedItem) -> str:
    """Human-readable raw_content for the synthetic ASSISTANT turn.

    The turn's parsed payload carries the real structured data. raw_content
    is the inspection-friendly version that surfaces in CLI session show
    and replay rendering. Marked explicitly as synthetic so a reader
    looking at the DB row knows it wasn't an LLM response.
    """
    return f"[synthetic retest turn reconstructed from learned_item {item.id}]"


def get_next_retest_turn(db: DbSession, session_id: str) -> ParsedTurn | None:
    """Return the next question for an in-progress retest.

    Walks the parent session's LearnedItems in created_at order and
    returns the first one not yet answered in the retest session.
    Returns None when all source items have been answered (caller
    surfaces session-end at that point).

    Read-only. The actual turn persistence happens when the user
    submits an answer.

    Raises RetestServiceError when session_id is not a retest
    session.
    """
    session = db.get(Session, session_id)
    if session is None:
        raise RetestServiceError("not_found", f"Session {session_id} not found")
    if session.parent_session_id is None:
        raise RetestServiceError(
            "not_eligible",
            f"Session {session_id} is not a retest session",
        )

    source_items = (
        db.execute(
            select(LearnedItem)
            .where(LearnedItem.session_id == session.parent_session_id)
            .order_by(LearnedItem.created_at.asc())
        )
        .scalars()
        .all()
    )

    # Count answered teaching turns (each ASSISTANT turn with
    # kind=turn paired with a following USER turn). The next item
    # to surface is at index `answered_count`.
    answered_count = _count_answered_questions(db, session_id)
    if answered_count >= len(source_items):
        return None
    return _reconstruct_parsed_turn(db, source_items[answered_count])


def _count_answered_questions(db: DbSession, session_id: str) -> int:
    """Count how many teaching turns in the retest have a paired user answer.

    Walks session turns in order. Each ASSISTANT turn with parsed
    kind="turn" followed by a USER turn counts as one answered
    question. Used by get_next_retest_turn to know where to resume.
    """
    turns = (
        db.execute(
            select(SessionTurn)
            .where(SessionTurn.session_id == session_id)
            .order_by(SessionTurn.turn_index.asc())
        )
        .scalars()
        .all()
    )

    count = 0
    expecting_user = False
    for turn in turns:
        if (
            turn.role is TurnRole.ASSISTANT
            and turn.parsed is not None
            and turn.parsed.get("kind") == "turn"
        ):
            expecting_user = True
            continue
        if turn.role is TurnRole.USER and expecting_user:
            count += 1
            expecting_user = False
    return count


async def grade_retest_answer(
    *,
    transport: LLMTransport[Any],
    question: str,
    expected_answer: str | None,
    user_answer: str,
) -> ParsedGrading:
    """Grade one retest answer via an LLM call on a fresh chat.

    Opens a new chat with the retest grading intro, sends the
    per-question payload, parses the response, closes the chat.
    Stateless: no chat handle is returned. Each call is
    independent.

    The transport is contacted directly. No DB writes happen
    here. The caller persists the resulting GRADING turn. This
    separation lets approve_session walk turns and pair them
    without the grading service needing to know about session
    structure.

    Raises RetestServiceError when:
    - the transport call fails (kind: "transport_failed")
    - the response is unparseable (kind: "parse_failed")
    - the response is the wrong shape, e.g. a teaching turn
      or a tool call (kind: "wrong_response_kind")

    Tool calls are explicitly rejected: the grading intro
    advertises no tools, but a misbehaving LLM might emit a
    TOOL_CALL block anyway. Treating it as a wrong-shape
    response keeps the grading call closed.
    """
    intro = build_retest_grading_intro()
    prompt = build_retest_grading_prompt(
        question=question,
        expected_answer=expected_answer,
        user_answer=user_answer,
    )

    try:
        chat, response = await transport.start_new_chat(intro, prompt)
    except TransportError as e:
        raise RetestServiceError(
            "transport_failed",
            f"Transport failed during retest grading: {e.message}",
        ) from e

    try:
        parsed = _response_to_parsed(response)
    except Exception as e:
        # Best-effort close before raising. If close fails too,
        # the original parse error is the more useful signal.
        with contextlib.suppress(TransportError):
            await transport.close(chat)
        raise RetestServiceError(
            "parse_failed",
            f"Parse failed on retest grading response: {e}",
        ) from e

    # Always close the chat before returning, success or otherwise.
    # If close fails the parsed grading is still useful. The close
    # failure is silently dropped.
    with contextlib.suppress(TransportError):
        await transport.close(chat)

    if isinstance(parsed, ParsedToolCall):
        tool_names = [c.name for c in parsed.calls]
        raise RetestServiceError(
            "wrong_response_kind",
            f"Retest grading response contained tool calls {tool_names!r}; "
            "the grading flow advertises no tools",
        )

    if not isinstance(parsed, ParsedGrading):
        raise RetestServiceError(
            "wrong_response_kind",
            f"Expected ParsedGrading from retest grading, got {parsed.kind!r}",
        )

    return parsed


def _response_to_parsed(response: TransportResponse) -> ParsedGrading | ParsedToolCall:
    """Translate a TransportResponse into a parsed shape for grading.

    Same dispatch as session_service._response_to_parsed: native
    tool_calls take precedence over text parsing. Returns
    ParsedGrading or ParsedToolCall (the only two shapes we
    accept here), other parsed kinds will fail in the caller's
    isinstance check and raise wrong_response_kind.

    Local to retest_service to avoid a cross-module import and
    because the dispatch is short enough that the duplication
    is the smaller cost.
    """
    if response.tool_calls:
        calls = list(response.tool_calls)
        raw_text = json.dumps([c.model_dump(mode="json") for c in calls])
        return ParsedToolCall(calls=calls, raw_text=raw_text)
    parsed = parse_response(response.text)
    if not isinstance(parsed, (ParsedGrading, ParsedToolCall)):
        # Parser produced something neither grading nor tool-call.
        # Wrap as a not-shaped-for-grading signal that the caller
        # converts to wrong_response_kind. We do not raise here
        # because the caller already has the isinstance check
        # and the error message matters at that layer.
        return parsed  # type: ignore[return-value]
    return parsed
