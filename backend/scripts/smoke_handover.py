"""Smoke test for the threshold-driven chat-handover flow.

Verifies that send_user_answer correctly splits the chat when the
session is at HANDOVER_THRESHOLD: the dying chat produces a handover
block, a new chat opens with that block seeded, and the user's
answer flows through the new chat. Five new turns persist on success.

Forces the message count to threshold rather than running 30 real
turns. This smoke tests the transition mechanism, not the
counting. The handover content will be thin since it summarizes
after one turn, but that does not affect what is being validated
here.

Run from backend/ with:

    uv run python scripts/smoke_handover.py
    uv run python scripts/smoke_handover.py --transport=playwright
    uv run python scripts/smoke_handover.py --all

Each run creates real rows in learning.db. Use `uv run python -m
cli.admin db reset` to clean up between runs.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import TYPE_CHECKING, Any, Literal

from app.core.config import get_settings
from app.core.db import SessionLocal
from app.models import SessionTurn, TransportKind, TurnRole
from app.schemas.parsed_response import ParsedTurn
from app.services.embedding_service import OpenRouterEmbedder
from app.services.session_service import (
    ESTIMATED_LOOKAHEAD_COST,
    HANDOVER_THRESHOLD,
    SessionServiceError,
    request_next_question,
    send_user_answer,
    start_session,
)
from app.transport import TransportError
from app.transport.deepseek_impl import DeepseekTransport
from app.transport.playwright_impl import PlaywrightClaudeTransport

if TYPE_CHECKING:
    from app.transport.base import LLMTransport


SMOKE_TOPIC_PATH = "Python > Data Types > Integers"
EXPECTED_NEW_TURNS = 5
EXPECTED_TURN_ROLES = (
    TurnRole.SYSTEM,
    TurnRole.ASSISTANT,
    TurnRole.TRANSITION,
    TurnRole.USER,
    TurnRole.ASSISTANT,
)

TransportChoice = Literal["deepseek", "playwright", "all"]


async def smoke_one(
    name: str,
    transport: LLMTransport[Any],
    transport_kind: TransportKind,
    embedder: OpenRouterEmbedder,
) -> None:
    """Run start + forced-threshold send and verify the handover flow."""
    print(f"=== {name} ===")
    print(f"  topic_path: {SMOKE_TOPIC_PATH}")
    print(f"  threshold: {HANDOVER_THRESHOLD}")

    with SessionLocal() as db:
        session, parsed = await start_session(
            db=db,
            transport=transport,
            transport_kind=transport_kind,
            topic_path=SMOKE_TOPIC_PATH,
            embedder=embedder,
        )
        print(f"  [start] session.id={session.id}")
        print(f"  [start] session.message_count={session.claude_chat_message_count}")
        original_chat_url = session.claude_chat_url

        # The user must answer first so the session reaches the
        # post-grading state that request_next_question requires.
        # We force the count to threshold just before request_next_question
        # so the handover fires there, not on send_user_answer.
        followup_answer = (
            parsed.expected_answer if parsed.expected_answer is not None else "I do not know."
        )
        print(f"  [send] answer={followup_answer[:60]!r}")
        grading = await send_user_answer(
            db=db,
            transport=transport,
            session_id=session.id,
            answer=followup_answer,
            embedder=embedder,
        )
        print(f"  [send] grading.kind={grading.kind}")

        # Force the count so request_next_question's look-ahead check
        # (current + ESTIMATED_LOOKAHEAD_COST > threshold) triggers
        # the handover path.
        forced_count = HANDOVER_THRESHOLD - ESTIMATED_LOOKAHEAD_COST + 1
        print(f"  [force] session.message_count := {forced_count}")
        session.claude_chat_message_count = forced_count
        db.commit()

        next_parsed = await request_next_question(
            db=db,
            transport=transport,
            session_id=session.id,
            embedder=embedder,
        )

        if not isinstance(next_parsed, ParsedTurn):
            raise RuntimeError(
                f"Expected ParsedTurn from new chat after handover, got {next_parsed.kind!r}.",
            )

        print(f"  [continue] new chat parsed.kind={next_parsed.kind}")
        print(f"  [continue] new chat parsed.mode={next_parsed.mode.value}")
        print(f"  [continue] new chat parsed.question={next_parsed.question[:100]!r}")

        db.refresh(session)
        print(f"  [continue] session.message_count={session.claude_chat_message_count}")
        if transport_kind is TransportKind.CLAUDE_PLAYWRIGHT:
            new_chat_url = session.claude_chat_url
            if new_chat_url == original_chat_url:
                raise RuntimeError("Playwright session.claude_chat_url unchanged across handover.")
            print(f"  [continue] new claude_chat_url={new_chat_url}")

        turns = (
            db.query(SessionTurn)
            .filter(SessionTurn.session_id == session.id)
            .order_by(SessionTurn.turn_index)
            .all()
        )
        # Pre-handover turns: SYSTEM(0), ASSISTANT(1) from start, plus
        # USER(2), GRADING(3) from the user's answer. The 5 transition
        # turns land at indexes 4..8.
        new_turns = turns[4:]
        if len(new_turns) != EXPECTED_NEW_TURNS:
            raise RuntimeError(
                f"Expected {EXPECTED_NEW_TURNS} new turns, got {len(new_turns)}.",
            )

        actual_roles = tuple(t.role for t in new_turns)
        if actual_roles != EXPECTED_TURN_ROLES:
            raise RuntimeError(
                f"Expected roles {EXPECTED_TURN_ROLES}, got {actual_roles}.",
            )

        print("  [check] 5 new turns in expected role order: OK")

        transition_turn = new_turns[2]
        if transition_turn.parsed is None:
            raise RuntimeError("TRANSITION turn has no parsed JSON.")

        handover = transition_turn.parsed
        print("  [transition] structured handover:")
        print(f"      DOMAIN_FOCUS:  {handover.get('domain_focus')!r}")
        print(f"      COVERED:       {handover.get('covered')!r}")
        print(f"      LAST_QUESTION: {handover.get('last_question')!r}")
        print(f"      NEXT_PLANNED:  {handover.get('next_planned')!r}")
        print(f"      OPEN_THREADS:  {handover.get('open_threads')!r}")
        print(f"      USER_STATE:    {handover.get('user_state')!r}")

        print(f"  [{name}] passed.\n")


async def run(choice: TransportChoice) -> None:
    """Dispatch to the chosen transport(s)."""
    settings = get_settings()

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
    parser = argparse.ArgumentParser(description="Handover-flow smoke test.")
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
