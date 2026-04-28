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
from app.services.session_service import SessionServiceError, start_session
from app.transport import TransportError
from app.transport.deepseek_impl import DeepseekTransport
from app.transport.playwright_impl import PlaywrightClaudeTransport

if TYPE_CHECKING:
    from app.transport.base import LLMTransport


SMOKE_TOPIC_PATH = "Python > Data Types > Integers"

TransportChoice = Literal["deepseek", "playwright", "all"]


async def smoke_one(name: str, transport: LLMTransport[Any]) -> None:
    """Run start_session once against the given transport."""
    print(f"=== {name} ===")
    print(f"  topic_path: {SMOKE_TOPIC_PATH}")

    with SessionLocal() as db:
        session, parsed = await start_session(
            db=db,
            transport=transport,
            topic_path=SMOKE_TOPIC_PATH,
        )

        print(f"  session.id={session.id}")
        print(f"  session.state={session.state.value}")
        print(f"  session.mode_used={session.mode_used.value}")
        print(f"  session.message_count={session.claude_chat_message_count}")
        print(f"  topic.id={session.topic_id}")
        print(f"  parsed.mode={parsed.mode.value}")
        print(f"  parsed.difficulty={parsed.difficulty.value}")
        print(f"  parsed.question={parsed.question[:100]!r}")
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
            await smoke_one(f"DeepSeek/{settings.deepseek_model}", ds)

    if choice in {"playwright", "all"}:
        print("Starting Playwright transport...\n")
        async with PlaywrightClaudeTransport(settings.chrome_profile_path) as pw:
            await smoke_one("Playwright/claude.ai", pw)

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
