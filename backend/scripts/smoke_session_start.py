"""Smoke test for the session-start flow against a real LLM.

Exercises start_session end to end: opens a real chat on the chosen
transport, sends the first prompt, parses the response, persists
the session and turns to the real DB.

By default runs against DeepSeek for speed. Pass --transport=playwright
to run against claude.ai, or --all to run against both transports
back-to-back.

Run from backend/ with:

    uv run python scripts/smoke_session_start.py
    uv run python scripts/smoke_session_start.py --transport=playwright
    uv run python scripts/smoke_session_start.py --all

Each run creates real rows in learning.db. Use `uv run python -m
cli.admin db reset` to clean up between runs if you want a fresh
slate, or `uv run python -m cli.admin db inspect` to see what was
written.

Requires:
  - DeepSeek path: DEEPSEEK_API_KEY in .env or process environment.
  - Playwright path: persistent Chrome profile logged in to claude.ai
    (run scripts/spike_claude_dom.py once if needed).
"""

from __future__ import annotations

import argparse
import asyncio
from typing import TYPE_CHECKING, Any, Literal

from app.core.config import get_settings
from app.core.db import SessionLocal
from app.models import LearnedItem, TransportKind
from app.services.session_service import (
    SessionServiceError,
    approve_session,
    send_user_answer,
    start_session,
)
from app.transport import TransportError
from app.transport.deepseek_impl import DeepseekTransport
from app.transport.playwright_impl import PlaywrightClaudeTransport

if TYPE_CHECKING:
    from app.transport.base import LLMTransport


SMOKE_TOPIC_PATH = "Python > Data Types > Integers"

TransportChoice = Literal["deepseek", "playwright", "all"]


async def smoke_one(
    name: str,
    transport: LLMTransport[Any],
    transport_kind: TransportKind,
) -> None:
    """Run start_session and one follow-up against the given transport."""
    print(f"=== {name} ===")
    print(f"  topic_path: {SMOKE_TOPIC_PATH}")

    with SessionLocal() as db:
        session, parsed = await start_session(
            db=db,
            transport=transport,
            transport_kind=transport_kind,
            topic_path=SMOKE_TOPIC_PATH,
        )

        print(f"  [start] session.id={session.id}")
        print(f"  [start] session.state={session.state.value}")
        print(f"  [start] session.mode_used={session.mode_used.value}")
        print(f"  [start] session.message_count={session.claude_chat_message_count}")
        print(f"  [start] topic.id={session.topic_id}")
        print(f"  [start] parsed.mode={parsed.mode.value}")
        print(f"  [start] parsed.difficulty={parsed.difficulty.value}")
        print(f"  [start] parsed.question={parsed.question[:100]!r}")

        followup_answer = (
            parsed.expected_answer if parsed.expected_answer is not None else "I do not know."
        )
        print(f"  [send] answer={followup_answer[:60]!r}")

        next_parsed = await send_user_answer(
            db=db,
            transport=transport,
            session_id=session.id,
            answer=followup_answer,
        )

        print(f"  [send] parsed.kind={next_parsed.kind}")
        if hasattr(next_parsed, "question"):
            print(f"  [send] parsed.question={next_parsed.question[:100]!r}")
        db.refresh(session)
        print(f"  [send] session.message_count={session.claude_chat_message_count}")

        completed = await approve_session(db=db, session_id=session.id)
        items = (
            db.query(LearnedItem)
            .filter(LearnedItem.session_id == completed.id)
            .order_by(LearnedItem.created_at)
            .all()
        )
        print(f"  [approve] session.state={completed.state.value}")
        print(f"  [approve] learned_items={len(items)}")
        for item in items:
            your_answer = item.your_answer or ""
            print(f"  [approve]   - {item.question[:60]!r} -> {your_answer[:40]!r}")
        print(f"  [{name}] passed.\n")


async def run(choice: TransportChoice) -> None:
    """Dispatch to the chosen transport(s)."""
    settings = get_settings()

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
            )

    if choice in {"playwright", "all"}:
        print("Starting Playwright transport...\n")
        async with PlaywrightClaudeTransport(settings.chrome_profile_path) as pw:
            await smoke_one(
                "Playwright/claude.ai",
                pw,
                TransportKind.CLAUDE_PLAYWRIGHT,
            )

    print("Smoke complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="5.1 session-service smoke test.")
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
