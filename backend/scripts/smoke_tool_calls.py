"""Smoke test for the tool-execution loop against real LLMs.

The LLM emits tool calls (either as ---TOOL_CALL--- blocks
on Claude transport or via native function calling on DeepSeek),
the registry executes handlers, and results flow back to the LLM.
Verifies the end-to-end path with real DB side effects.

Uses a topic in a non-existent domain to nudge the LLM toward
calling create_domain. If the LLM complies, a Domain row lands
in the DB. If it doesn't comply, the smoke fails and we know
the intro needs more work to motivate tool use.

Run from backend/ with:

    uv run python scripts/smoke_tool_calls.py
    uv run python scripts/smoke_tool_calls.py --transport=playwright
    uv run python scripts/smoke_tool_calls.py --all

Each run creates real rows in learning.db. Use `uv run python -m
cli.admin db reset` between runs if you want a fresh slate.

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
from app.models import Domain, SessionTurn, TransportKind, TurnRole
from app.services.session_service import (
    SessionServiceError,
    send_user_answer,
    start_session,
)
from app.transport import TransportError
from app.transport.deepseek_impl import DeepseekTransport
from app.transport.playwright_impl import PlaywrightClaudeTransport

if TYPE_CHECKING:
    from app.transport.base import LLMTransport


# An obscure-enough language that it won't be in any seed list.
# If the seed list ever includes Mojo, switch to another fresh
# domain name. The point is: this domain must not exist in the
# DB before the smoke runs, so create_domain has a reason to fire.
SMOKE_DOMAIN_NAME = "Mojo"
SMOKE_TOPIC_PATH = f"{SMOKE_DOMAIN_NAME} > Memory Model > Ownership"

# A follow-up question that nudges the LLM to look up existing
# topics in the same domain. Designed to make get_topics_by_domain
# a useful next move.
FOLLOWUP_ANSWER = (
    "Before answering, please tell me what other Mojo topics "
    "exist in the system so I can plan my study order."
)

TransportChoice = Literal["deepseek", "playwright", "all"]


async def smoke_one(
    name: str,
    transport: LLMTransport[Any],
    transport_kind: TransportKind,
) -> None:
    """Run start + follow-up against the given transport.

    Two empirical checks:
      1. Domain row for Mojo exists after the run (proves
         create_domain handler ran).
      2. At least one TOOL_CALL turn exists on the session
         (proves the helper persisted a tool-call turn pair
         during the follow-up).

    Either check passing means the path works. Both failing
    means the LLM did not use tools at all, which is a real
    finding worth investigating.
    """
    print(f"=== {name} ===")
    print(f"  topic_path: {SMOKE_TOPIC_PATH}")

    with SessionLocal() as db:
        # Confirm clean starting state for the smoke domain.
        existing_mojo = db.query(Domain).filter(Domain.name == SMOKE_DOMAIN_NAME).one_or_none()
        if existing_mojo is not None:
            print(
                f"  [warn] {SMOKE_DOMAIN_NAME!r} domain already exists. "
                f"Run `uv run python -m cli.admin db reset` for a clean smoke."
            )

        session, parsed = await start_session(
            db=db,
            transport=transport,
            transport_kind=transport_kind,
            topic_path=SMOKE_TOPIC_PATH,
        )
        print(f"  [start] session.id={session.id}")
        print(f"  [start] session.state={session.state.value}")
        print(f"  [start] parsed.question={parsed.question[:100]!r}")

        # Check 1: did the LLM call create_domain during start?
        # Tool calls during start_session do not persist as turns
        # (session_id=None in the helper) but their DB side effects
        # do persist.
        post_start_mojo = db.query(Domain).filter(Domain.name == SMOKE_DOMAIN_NAME).one_or_none()
        domain_created_at_start = post_start_mojo is not None and existing_mojo is None
        if domain_created_at_start:
            print(f"  [check] {SMOKE_DOMAIN_NAME!r} Domain row created during start: OK")
        else:
            print(
                f"  [check] {SMOKE_DOMAIN_NAME!r} Domain row not created during start. "
                f"LLM may have skipped create_domain."
            )

        print(f"  [send] answer={FOLLOWUP_ANSWER[:80]!r}")

        next_parsed = await send_user_answer(
            db=db,
            transport=transport,
            session_id=session.id,
            answer=FOLLOWUP_ANSWER,
        )
        print(f"  [send] parsed.kind={next_parsed.kind}")

        # Check 2: did the helper persist a TOOL_CALL turn during
        # the follow-up?
        tool_call_turns = (
            db.query(SessionTurn)
            .filter(SessionTurn.session_id == session.id)
            .filter(SessionTurn.role == TurnRole.TOOL_CALL)
            .all()
        )
        tool_result_turns = (
            db.query(SessionTurn)
            .filter(SessionTurn.session_id == session.id)
            .filter(SessionTurn.role == TurnRole.TOOL_RESULT)
            .all()
        )

        if tool_call_turns:
            print(f"  [check] TOOL_CALL turns persisted: {len(tool_call_turns)}")
            for turn in tool_call_turns:
                if turn.parsed is not None and "call" in turn.parsed:
                    call = turn.parsed["call"]
                    print(f"  [check]   - tool: {call.get('name')!r} args: {call.get('args')}")
            if len(tool_result_turns) != len(tool_call_turns):
                raise RuntimeError(
                    f"Mismatched TOOL_CALL/TOOL_RESULT turn counts: "
                    f"{len(tool_call_turns)} calls, {len(tool_result_turns)} results."
                )
        else:
            print("  [check] No TOOL_CALL turns persisted during follow-up.")

        if not domain_created_at_start and not tool_call_turns:
            raise RuntimeError(
                f"Neither check passed: no Domain row and no TOOL_CALL turns. "
                f"The LLM did not exercise the tool path. "
                f"Inspect the chat by querying session_turn for session_id={session.id!r}."
            )

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
    parser = argparse.ArgumentParser(description="Tool-execution loop smoke test.")
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
