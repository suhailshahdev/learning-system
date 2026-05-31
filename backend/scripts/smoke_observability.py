"""Smoke test for LLM observability: llm_call rows and trace linkage.

Verifies the observability path end to end against a real LLM:
that the instrumented transport writes an llm_call row per round-trip
with sane values, and that an error logged during a turn shares its
trace id with that turn's llm_call rows.

Unlike the older smokes, this one builds the production observability
stack itself: it wraps the bare transport in InstrumentedTransport
with a WritingRecorder bound to SessionLocal, the same wiring the
app lifespan uses. The older smokes drive a bare transport because
previously there was nothing to record.

Following the smoke convention, the row contents are reported as
evidence rather than asserted field-by-field. Two things are checked
hard because they are deterministic, not LLM-dependent: that rows
landed at all, and that the trace ids match. An LLM misbehaving
cannot affect either.

The genuine-LLM-parse-failure path (a real model emits malformed
output, the service logs a parse error) is covered by the unchanged
transport-error handling, not re-proven here. This smoke forces the
linkage deterministically by logging an error inside the same turn
trace as a real successful round-trip, which is what exercises the
shared-contextvar mechanism this commit adds.

Run from backend/ with:

    uv run python scripts/smoke_observability.py
    uv run python scripts/smoke_observability.py --transport=playwright
    uv run python scripts/smoke_observability.py --all

Each run creates real rows in learning.db (sessions, turns, llm_call,
and one error_log row). Use `uv run python -m cli.admin db reset` to
clean up between runs.
"""

from __future__ import annotations

import argparse
import asyncio
from typing import TYPE_CHECKING, Any, Literal

from app.core.config import get_settings
from app.core.db import SessionLocal
from app.core.trace_context import turn_trace
from app.models import ErrorLog, LLMCall, TransportKind
from app.services.embedding_service import OpenRouterEmbedder
from app.services.llm_call_recorder import WritingRecorder
from app.services.session_service import (
    SessionServiceError,
    _log_service_error,
    send_user_answer,
    start_session,
)
from app.transport import TransportError
from app.transport.deepseek_impl import DeepseekTransport
from app.transport.instrumented import InstrumentedTransport
from app.transport.playwright_impl import PlaywrightClaudeTransport
from sqlalchemy import select

if TYPE_CHECKING:
    from app.transport.base import LLMTransport


SMOKE_TOPIC_PATH = "Python > Data Types > Integers"

TransportChoice = Literal["deepseek", "playwright", "all"]


async def smoke_one(
    name: str,
    bare_transport: LLMTransport[Any],
    transport_kind: TransportKind,
    model: str | None,
    embedder: OpenRouterEmbedder,
) -> None:
    """Drive a real turn through the instrumented stack and check rows."""
    print(f"=== {name} ===")
    print(f"  topic_path: {SMOKE_TOPIC_PATH}")

    recorder = WritingRecorder(SessionLocal)
    transport: LLMTransport[Any] = InstrumentedTransport(
        bare_transport, recorder, transport_kind, model=model
    )

    with SessionLocal() as db:
        session, parsed = await start_session(
            db=db,
            transport=transport,
            transport_kind=transport_kind,
            topic_path=SMOKE_TOPIC_PATH,
            embedder=embedder,
        )
        print(f"  [start] session.id={session.id}")

        followup_answer = (
            parsed.expected_answer if parsed.expected_answer is not None else "I do not know."
        )
        grading = await send_user_answer(
            db=db,
            transport=transport,
            session_id=session.id,
            answer=followup_answer,
            embedder=embedder,
        )
        print(f"  [send] grading.kind={grading.kind}")

    # --- Check 1: llm_call rows landed and look sane ---
    # WritingRecorder commits on its own short sessions, so open a
    # fresh one to read them back.
    with SessionLocal() as db:
        calls = list(db.execute(select(LLMCall)).scalars().all())

    if not calls:
        raise RuntimeError(
            "No llm_call rows written. The instrumented transport or "
            "recorder is not persisting. This is deterministic, not an "
            "LLM issue.",
        )

    print(f"  [check] llm_call rows written: {len(calls)}")
    for call in calls:
        print(
            f"  [llm_call]   {call.transport_kind.value}.{call.method} "
            f"latency={call.latency_ms}ms "
            f"prompt_chars={call.prompt_chars} response_chars={call.response_chars} "
            f"success={call.success} "
            f"tokens={call.prompt_tokens}/{call.completion_tokens} "
            f"trace={call.trace_id[:12]}..."
        )

    distinct_traces = {c.trace_id for c in calls}
    print(f"  [check] distinct trace ids across calls: {len(distinct_traces)}")

    # --- Check 2: an error logged inside a turn trace shares the
    # turn's trace id. Drive one real round-trip inside an explicit
    # turn_trace, then log an error in the same context, and assert
    # both the llm_call row and the error_log row carry turn_id.
    # Both lookups key on turn_id (unique per run) so prior runs'
    # probe rows in a shared learning.db do not match. ---
    with turn_trace() as turn_id:
        # A real round-trip inside this turn trace: its llm_call row
        # will carry turn_id (the wrapper reads the contextvar first).
        chat, _ = await transport.start_new_chat("Smoke linkage probe.", "Say OK.")
        await transport.close(chat)

        # An error logged in the same turn trace must carry turn_id too.
        with SessionLocal() as db:
            _log_service_error(
                db,
                kind="smoke.observability.linkage_probe",
                message="Deliberate probe to verify error-to-call trace linkage.",
                session_id=None,
            )

    with SessionLocal() as db:
        probe_call = (
            db.execute(select(LLMCall).where(LLMCall.trace_id == turn_id)).scalars().first()
        )
        probe_error = (
            db.execute(select(ErrorLog).where(ErrorLog.trace_id == turn_id)).scalars().first()
        )

    if probe_call is None:
        raise RuntimeError(
            f"No llm_call row carries the turn trace id {turn_id[:12]}.... "
            "The wrapper is not reading the turn context.",
        )
    if probe_error is None:
        raise RuntimeError("Probe error_log row not found.")
    if probe_error.trace_id != turn_id:
        raise RuntimeError(
            f"Linkage broken: error_log.trace_id={probe_error.trace_id!r} "
            f"!= turn trace id {turn_id!r}.",
        )

    print(f"  [check] error-to-call linkage: error and call share trace {turn_id[:12]}... OK")
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
                    settings.deepseek_model,
                    embedder,
                )

        if choice in {"playwright", "all"}:
            print("Starting Playwright transport...\n")
            async with PlaywrightClaudeTransport(settings.chrome_profile_path) as pw:
                await smoke_one(
                    "Playwright/claude.ai",
                    pw,
                    TransportKind.CLAUDE_PLAYWRIGHT,
                    None,
                    embedder,
                )

    print("Smoke complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="M13 observability smoke test.")
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
