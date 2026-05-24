"""Smoke test for the diagnostic-mode flow against a real LLM.

Exercises propose_topic end to end: seeds learned items with mixed
grading verdicts and stale topics, opens a real chat on the chosen
transport, lets the LLM call analytical tools, parses the PROPOSAL
response.

The seeded data is designed to produce a meaningful proposal: one
topic with mostly-incorrect attempts (should surface in
get_weak_topics), one topic with old last_reviewed_at (should
surface in get_stale_topics), one clean topic that should NOT be
proposed.

This is a soft fail smoke test. Hard failures like parse errors,
wrong response kind, and transport errors surface as exceptions.
Soft failures like a nonsensical topic, vague reasoning, or the
LLM ignoring seeded data need human review. The script prints the
seeded data, tool calls made, and final proposal so eyeballing is
straightforward.

By default runs against DeepSeek pro. Pass --transport=playwright
for claude.ai, or --all for both.

Run from backend/ with:

    uv run python scripts/smoke_diagnostic.py
    uv run python scripts/smoke_diagnostic.py --transport=playwright
    uv run python scripts/smoke_diagnostic.py --all

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
from app.services.diagnostic_service import (
    DiagnosticServiceError,
    propose_topic,
)
from app.services.embedding_service import OpenRouterEmbedder
from app.transport import TransportError
from app.transport.deepseek_impl import DeepseekTransport
from app.transport.playwright_impl import PlaywrightClaudeTransport

if TYPE_CHECKING:
    from app.transport.base import LLMTransport
    from sqlalchemy.orm import Session as DbSession


# Seeded topics. The diagnostic LLM should propose one of the first
# two (weak or stale) and avoid the third (clean). The exact choice
# is the LLM's and the smoke doesn't enforce which.
WEAK_TOPIC = "Python > Data Types > Integers"
STALE_TOPIC = "Python > Concepts > Decorators"
CLEAN_TOPIC = "Python > Basics > Variables"

TransportChoice = Literal["deepseek", "playwright", "all"]


def _ensure_topic(db: DbSession, path: str, last_reviewed_at: datetime | None) -> Topic:
    """Get-or-create a Topic at the given path with the given timestamp.

    Mirrors the topic_crud.get_or_create_topic pattern but with the
    last_reviewed_at override the smoke needs. Avoids depending on
    a separate seed path.
    """
    existing = db.query(Topic).filter(Topic.path == path).one_or_none()
    if existing is not None:
        existing.last_reviewed_at = last_reviewed_at
        return existing
    topic = Topic(
        path=path,
        domain=path.split(" > ", 1)[0],
        name=path.rsplit(" > ", 1)[-1],
        status=TopicStatus.IN_PROGRESS,
        last_reviewed_at=last_reviewed_at,
    )
    db.add(topic)
    db.flush()
    return topic


def _seed_smoke_data(db: DbSession) -> None:
    """Seed the DB with synthetic learned items for the smoke run.

    Three topics with distinct shapes:
      - WEAK_TOPIC: 5 attempts, 4 incorrect, 1 correct, reviewed today
      - STALE_TOPIC: 3 attempts, all correct, reviewed 60 days ago
      - CLEAN_TOPIC: 3 attempts, all correct, reviewed today

    Creates one synthetic Session row to attach the learned items to,
    since LearnedItem.session_id is a foreign key.
    """
    now = datetime.now(UTC)
    sixty_days_ago = now - timedelta(days=60)

    weak = _ensure_topic(db, WEAK_TOPIC, last_reviewed_at=now)
    stale = _ensure_topic(db, STALE_TOPIC, last_reviewed_at=sixty_days_ago)
    clean = _ensure_topic(db, CLEAN_TOPIC, last_reviewed_at=now)

    # One synthetic session to anchor the learned items.
    session = Session(
        topic_id=None,
        mode_used=LearningMode.FLASHCARD,
        state=SessionState.COMPLETED,
        transport_kind=TransportKind.DEEPSEEK,
        claude_chat_url=None,
        claude_chat_message_count=0,
        active_preferences=[],
        context_snapshot={},
    )
    db.add(session)
    db.flush()

    weak_questions = [
        ("What is 7 // 2 in Python 3?", GradingVerdict.INCORRECT),
        ("What does -7 // 2 evaluate to?", GradingVerdict.INCORRECT),
        ("What is 7 % -2?", GradingVerdict.INCORRECT),
        ("Does Python distinguish int from long?", GradingVerdict.INCORRECT),
        ("What is bool(0)?", GradingVerdict.CORRECT),
    ]
    for question, verdict in weak_questions:
        db.add(
            LearnedItem(
                session_id=session.id,
                topic_id=weak.id,
                question=question,
                answer="canonical",
                your_answer="user_answer",
                mode=LearningMode.FLASHCARD,
                difficulty=Difficulty.BEGINNER,
                grading_verdict=verdict,
                status=LearnedItemStatus.LEARNED,
                last_reviewed_at=now,
            )
        )

    stale_questions = [
        "What is a decorator?",
        "Show a decorator that times a function.",
        "What does functools.wraps do?",
    ]
    for question in stale_questions:
        db.add(
            LearnedItem(
                session_id=session.id,
                topic_id=stale.id,
                question=question,
                answer="canonical",
                your_answer="user_answer",
                mode=LearningMode.FLASHCARD,
                difficulty=Difficulty.INTERMEDIATE,
                grading_verdict=GradingVerdict.CORRECT,
                status=LearnedItemStatus.LEARNED,
                last_reviewed_at=sixty_days_ago,
            )
        )

    clean_questions = [
        "What is a variable in Python?",
        "How do you assign a value?",
        "What is type inference?",
    ]
    for question in clean_questions:
        db.add(
            LearnedItem(
                session_id=session.id,
                topic_id=clean.id,
                question=question,
                answer="canonical",
                your_answer="user_answer",
                mode=LearningMode.FLASHCARD,
                difficulty=Difficulty.BEGINNER,
                grading_verdict=GradingVerdict.CORRECT,
                status=LearnedItemStatus.LEARNED,
                last_reviewed_at=now,
            )
        )

    db.commit()


def _print_seeded_summary() -> None:
    """Print what was seeded so the human reading smoke output knows.

    Without this, the proposal lands without context and "is this
    reasonable" becomes hard to judge.
    """
    print("Seeded data summary:")
    print(f"  WEAK:  {WEAK_TOPIC}")
    print("    5 attempts: 4 incorrect, 1 correct, reviewed today")
    print(f"  STALE: {STALE_TOPIC}")
    print("    3 attempts: 3 correct, reviewed 60 days ago")
    print(f"  CLEAN: {CLEAN_TOPIC}")
    print("    3 attempts: 3 correct, reviewed today")
    print("  Expectation: LLM proposes WEAK or STALE, avoids CLEAN.")
    print()


async def smoke_one(
    name: str,
    transport: LLMTransport[Any],
    transport_kind: TransportKind,
    embedder: OpenRouterEmbedder,
) -> None:
    """Run propose_topic against the given transport and print results."""
    print(f"=== {name} ===")

    with SessionLocal() as db:
        _seed_smoke_data(db)

        proposal = await propose_topic(
            db=db,
            transport=transport,
            transport_kind=transport_kind,
            embedder=embedder,
        )

        print(f"  [proposal] topic_path: {proposal.topic_path}")
        print(f"  [proposal] reasoning:  {proposal.reasoning}")

        # Eyeball check: did the LLM propose one of the seeded topics
        # we expected, or something else? Print a verdict but do not
        # raise because the LLM might have legitimate reasons to propose
        # something else (e.g., the seeded data is one slice of state).
        if proposal.topic_path == WEAK_TOPIC:
            print("  [check] LLM proposed WEAK topic. Expected outcome.")
        elif proposal.topic_path == STALE_TOPIC:
            print("  [check] LLM proposed STALE topic. Also reasonable.")
        elif proposal.topic_path == CLEAN_TOPIC:
            print("  [check] LLM proposed CLEAN topic. Soft fail: clean topic should not surface.")
        else:
            print(f"  [check] LLM proposed unseeded topic: {proposal.topic_path!r}.")
            print("           Soft fail unless the LLM has a clear reason in its reasoning.")

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
    parser = argparse.ArgumentParser(description="Diagnostic-mode smoke test.")
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
    except DiagnosticServiceError as e:
        print(f"\nDiagnostic service error: {e.message}")
        print(f"Kind: {e.kind}")
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
