"""Smoke test for the retest flow against a real LLM.

Exercises start_retest → answer/grade loop → ParsedSessionEnd →
approve end to end. Seeds a synthetic COMPLETED source session
with three LearnedItems spanning verdict diversity (clearly
correct, clearly wrong, open-graded), runs a retest of it, feeds
deliberate answers, prints each grading for human review, and
verifies the parent's items got last_reviewed_at bumped on
approve.

The retest grading path runs fresh-chat-per-question. For
three questions, that's three round trips.

Hard failures (exceptions): start eligibility, transport errors,
parse errors, wrong-response-kind, approve errors, missing
last_reviewed_at bump. Soft failures need a human read: a
clearly-wrong answer graded `correct`, a clearly-correct answer
graded `incorrect`, open-graded explanation that's vacuous, or
the grading intro misfiring with a tool call.

By default runs against DeepSeek pro. Pass --transport=playwright
for claude.ai, or --all for both.

Run from backend/ with:

    uv run python scripts/smoke_retest.py
    uv run python scripts/smoke_retest.py --transport=playwright
    uv run python scripts/smoke_retest.py --all

Each run creates real rows in learning.db. Use `uv run python -m
cli.admin db reset` to clean up between runs.

Requires:
  - DeepSeek path: DEEPSEEK_API_KEY in .env or process environment.
  - Playwright path: persistent Chrome profile logged in to claude.ai.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal

from app.core.config import get_settings
from app.core.db import SessionLocal
from app.models import (
    Difficulty,
    GradingVerdict,
    LearnedItem,
    LearnedItemStatus,
    LearningMode,
    Session,
    SessionState,
    Topic,
    TopicStatus,
    TransportKind,
)
from app.schemas.parsed_response import ParsedGrading, ParsedSessionEnd, ParsedTurn
from app.services.embedding_service import OpenRouterEmbedder
from app.services.retest_service import RetestServiceError, start_retest
from app.services.session_service import (
    OPEN_ANSWER_PLACEHOLDER,
    SessionServiceError,
    approve_session,
    request_next_question,
    send_user_answer,
)
from app.transport import TransportError
from app.transport.deepseek_impl import DeepseekTransport
from app.transport.playwright_impl import PlaywrightClaudeTransport

if TYPE_CHECKING:
    from app.transport.base import LLMTransport
    from sqlalchemy.orm import Session as DbSession


# The retest source. Three items spanning verdict diversity:
# - Q1 has canonical answer, we'll send the right one (expect correct)
# - Q2 has canonical answer, we'll send a wrong one (expect incorrect)
# - Q3 is open-graded, we'll send a substantive answer (expect open_graded)
RETEST_TOPIC = "Python > Data Types > Lists"

SOURCE_QUESTIONS: list[tuple[str, str, LearningMode, str, str]] = [
    # (question, canonical_answer, mode, our_retest_answer, expected_verdict_for_eyeball)
    (
        "What method appends a single element to the end of a list?",
        "append()",
        LearningMode.TYPE_THE_ANSWER,
        "append()",
        "correct",
    ),
    (
        "What does list.pop() with no argument return?",
        "The last element of the list, which is also removed.",
        LearningMode.TYPE_THE_ANSWER,
        "It returns the first element and shifts everything left.",
        "incorrect",
    ),
    (
        "Explain in your own words why list slicing creates a new list rather than a view.",
        # OPEN sentinel survived through approve as OPEN_ANSWER_PLACEHOLDER.
        OPEN_ANSWER_PLACEHOLDER,
        LearningMode.EXPLAIN_BACK,
        (
            "Slicing returns a new list object holding references to the "
            "same elements, so mutating the slice doesn't affect the "
            "original. This is different from NumPy slicing, which "
            "returns a view."
        ),
        "open_graded",
    ),
]

TransportChoice = Literal["deepseek", "playwright", "all"]


def _ensure_topic(db: DbSession, path: str) -> Topic:
    """Get-or-create a Topic at the given path with no review timestamp."""
    existing = db.query(Topic).filter(Topic.path == path).one_or_none()
    if existing is not None:
        return existing
    topic = Topic(
        path=path,
        domain=path.split(" > ", 1)[0],
        name=path.rsplit(" > ", 1)[-1],
        status=TopicStatus.IN_PROGRESS,
        last_reviewed_at=None,
    )
    db.add(topic)
    db.flush()
    return topic


def _seed_source_session(db: DbSession) -> Session:
    """Seed a COMPLETED source session with three LearnedItems.

    Items are timestamped a few days back so the approve-bump on
    parent items is observably newer afterwards. created_at on
    LearnedItem drives retest ordering (created_at asc), so the
    items are inserted in the order we want the retest to
    walk them.

    Returns the source session for retest start.
    """
    topic = _ensure_topic(db, RETEST_TOPIC)
    a_few_days_ago = datetime.now(UTC) - timedelta(days=3)

    session = Session(
        topic_id=topic.id,
        mode_used=LearningMode.TYPE_THE_ANSWER,
        state=SessionState.COMPLETED,
        transport_kind=TransportKind.DEEPSEEK,
        claude_chat_url=None,
        claude_chat_message_count=0,
        active_preferences=[],
        context_snapshot={},
    )
    db.add(session)
    db.flush()

    for question, canonical_answer, mode, _retest_answer, _expected in SOURCE_QUESTIONS:
        db.add(
            LearnedItem(
                session_id=session.id,
                topic_id=topic.id,
                question=question,
                answer=canonical_answer,
                your_answer="<source-session user answer, not used by retest>",
                mode=mode,
                difficulty=Difficulty.BEGINNER,
                grading_verdict=GradingVerdict.CORRECT,
                status=LearnedItemStatus.LEARNED,
                last_reviewed_at=a_few_days_ago,
            )
        )

    db.commit()
    db.refresh(session)
    return session


def _print_seeded_summary() -> None:
    """Print what's seeded so the human reading smoke output knows."""
    print("Seeded source session for retest:")
    print(f"  Topic: {RETEST_TOPIC}")
    for i, (question, canonical, _mode, our_answer, expected) in enumerate(SOURCE_QUESTIONS, 1):
        print(f"  Q{i}: {question}")
        if canonical == OPEN_ANSWER_PLACEHOLDER:
            print("     canonical: (open-graded)")
        else:
            print(f"     canonical: {canonical}")
        print(f"     we'll send: {our_answer}")
        print(f"     eyeball-expect: {expected}")
    print("  Each source item last_reviewed_at is ~3 days old; approve will bump them to now.\n")


def _eyeball_verdict(actual: GradingVerdict, expected_label: str) -> str:
    """Return a short eyeball-check line comparing actual to expectation."""
    if expected_label == "correct" and actual is GradingVerdict.CORRECT:
        return "matches expectation."
    if expected_label == "incorrect" and actual is GradingVerdict.INCORRECT:
        return "matches expectation."
    if expected_label == "open_graded" and actual is GradingVerdict.OPEN_GRADED:
        return "matches expectation."
    if expected_label == "correct" and actual is GradingVerdict.PARTIAL:
        return "PARTIAL on a clearly-correct answer; lenient but acceptable."
    if expected_label == "incorrect" and actual is GradingVerdict.PARTIAL:
        return "PARTIAL on a clearly-wrong answer; lenient but acceptable."
    return f"SOFT FAIL: expected {expected_label}, got {actual.value}. Review explanation."


async def _walk_retest(
    db: DbSession,
    transport: LLMTransport[Any],
    transport_kind: TransportKind,
    source_session: Session,
    embedder: OpenRouterEmbedder,
) -> Session:
    """Run a full retest cycle: start, loop answer/next, return retest session.

    Each iteration: read the current ParsedTurn from the most recent
    teaching turn (the first one comes from start_retest, subsequent
    ones from request_next_question), send the answer we prepared,
    print the grading, ask for the next question. Loop ends when
    request_next_question returns a ParsedSessionEnd.
    """
    source_id = source_session.id

    print("Starting retest...")
    retest_session, first_turn = start_retest(
        db,
        source_session_id=source_id,
        transport_kind=transport_kind,
    )
    print(f"  retest session: {retest_session.id}")
    print(f"  first question: {first_turn.question}\n")

    current_turn: ParsedTurn = first_turn
    for i, (_question, _canonical, _mode, our_answer, expected) in enumerate(SOURCE_QUESTIONS, 1):
        print(f"  Q{i}: {current_turn.question}")
        print(f"     answer being sent: {our_answer}")

        grading_response = await send_user_answer(
            db=db,
            transport=transport,
            session_id=retest_session.id,
            answer=our_answer,
            embedder=embedder,
        )
        # The retest grading flow always returns ParsedGrading.
        # ParsedSessionEnd lands from request_next_question, not here.
        # ParsedToolCall is rejected in retest_service as wrong_response_kind.
        if not isinstance(grading_response, ParsedGrading):
            raise RuntimeError(
                f"Expected ParsedGrading on retest answer, got {type(grading_response).__name__} "
                f"with kind={grading_response.kind!r}."
            )

        print(f"     verdict: {grading_response.verdict.value}")
        print(f"     explanation: {grading_response.explanation}")
        if grading_response.explanation_code is not None:
            print(
                f"     explanation_code ({grading_response.explanation_code.language}): "
                f"{grading_response.explanation_code.body[:120]}..."
            )
        print(f"     [check] {_eyeball_verdict(grading_response.verdict, expected)}\n")

        # Ask for the next question. The last iteration should land
        # ParsedSessionEnd, earlier iterations land another ParsedTurn.
        next_response = await request_next_question(
            db=db,
            transport=transport,
            session_id=retest_session.id,
            embedder=embedder,
        )

        is_last = i == len(SOURCE_QUESTIONS)
        if is_last:
            if not isinstance(next_response, ParsedSessionEnd):
                raise RuntimeError(
                    "Expected ParsedSessionEnd after exhausting source items, "
                    f"got {type(next_response).__name__} with kind={next_response.kind!r}."
                )
            print(f"  source exhausted; session-end summary: {next_response.summary}\n")
        else:
            if not isinstance(next_response, ParsedTurn):
                raise RuntimeError(
                    f"Expected ParsedTurn for next retest question, got "
                    f"{type(next_response).__name__} with kind={next_response.kind!r}."
                )
            current_turn = next_response

    return retest_session


def _assert_parent_items_bumped(
    db: DbSession,
    source_session_id: str,
    before_bump_threshold: datetime,
) -> None:
    """Verify that source items' last_reviewed_at was bumped past the threshold.

    approve_session on a retest bumps the parent's LearnedItems'
    last_reviewed_at to `now`, inside the same transaction as the
    minted retest items and derived assertions. Threshold is captured
    just before approve fires so we don't get a false pass from
    timestamps that pre-date the call.
    """
    source_items = db.query(LearnedItem).filter(LearnedItem.session_id == source_session_id).all()
    if not source_items:
        raise RuntimeError(f"Source session {source_session_id} has no items to check.")

    # LearnedItem.last_reviewed_at is nullable on the model. Items
    # missing the bump (still None) and items whose bump didn't
    # advance past the threshold both count as stale here.
    stale = [
        item
        for item in source_items
        if item.last_reviewed_at is None or item.last_reviewed_at <= before_bump_threshold
    ]
    if stale:
        ids = [item.id for item in stale]
        raise RuntimeError(
            f"D394 violated: {len(stale)} source LearnedItems not bumped past "
            f"approve threshold. Item ids: {ids}."
        )
    print(
        f"  [check] all {len(source_items)} source items bumped past "
        f"{before_bump_threshold.isoformat()}."
    )


async def smoke_one(
    name: str,
    transport: LLMTransport[Any],
    transport_kind: TransportKind,
    embedder: OpenRouterEmbedder,
) -> None:
    """Run the full retest smoke against the given transport."""
    print(f"=== {name} ===")

    with SessionLocal() as db:
        source = _seed_source_session(db)
        source_id = source.id

        retest = await _walk_retest(db, transport, transport_kind, source, embedder)

        # Capture the threshold just before approve. Bump
        # has to land after this.
        before_approve = datetime.now(UTC)
        print("Approving retest...")
        approved = await approve_session(db=db, session_id=retest.id)
        print(f"  retest state after approve: {approved.state.value}")

        # Read the source items fresh from DB. The approve transaction
        # has committed, so plain query returns up-to-date rows.
        _assert_parent_items_bumped(db, source_id, before_approve)

        # Count minted items on the retest session itself.
        child_items = db.query(LearnedItem).filter(LearnedItem.session_id == retest.id).all()
        print(f"  [check] retest minted {len(child_items)} new LearnedItems on child session.")
        if len(child_items) != len(SOURCE_QUESTIONS):
            raise RuntimeError(
                f"Expected {len(SOURCE_QUESTIONS)} child items, got {len(child_items)}."
            )

        print(f"  [{name}] hard-checks passed.\n")


async def run(choice: TransportChoice) -> None:
    """Dispatch to the chosen transport(s)."""
    settings = get_settings()

    _print_seeded_summary()

    async with OpenRouterEmbedder(
        api_key=settings.openrouter_api_key.get_secret_value(),
        model=settings.openrouter_embedding_model,
    ) as embedder:
        if choice in {"deepseek", "all"}:
            print("Starting DeepSeek transport...\n")
            async with DeepseekTransport(
                api_key=settings.deepseek_api_key.get_secret_value(),
                default_model=settings.deepseek_model,
            ) as ds:
                await smoke_one(
                    f"DeepSeek/{settings.deepseek_model}",
                    ds,
                    TransportKind.DEEPSEEK,
                    embedder,
                )

        if choice in {"playwright", "all"}:
            print("Starting Playwright transport...\n")
            async with PlaywrightClaudeTransport(settings.chrome_profile_path) as pw:
                await smoke_one(
                    "Playwright/claude.ai",
                    pw,
                    TransportKind.CLAUDE_PLAYWRIGHT,
                    embedder,
                )

    print("Smoke complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Retest-flow smoke test.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--transport",
        choices=["deepseek", "playwright"],
        default="deepseek",
        help="Which transport to run against (default: deepseek).",
    )
    group.add_argument(
        "--all",
        action="store_const",
        const="all",
        dest="transport",
        help="Run against both transports.",
    )
    args = parser.parse_args()

    try:
        asyncio.run(run(args.transport))
    except RetestServiceError as e:
        print(f"\nRetest service error: {e}")
        print(f"Kind: {e.kind}")
        raise SystemExit(1) from e
    except SessionServiceError as e:
        print(f"\nSession service error: {e.message}")
        if e.cause is not None:
            print(f"Caused by: {type(e.cause).__name__}: {e.cause}")
        raise SystemExit(1) from e
    except TransportError as e:
        print(f"\nTransport error: {e.message}")
        if e.cause is not None:
            print(f"Caused by: {type(e.cause).__name__}: {e.cause}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
