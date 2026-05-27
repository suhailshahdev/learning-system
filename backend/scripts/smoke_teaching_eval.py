"""Smoke test for the teaching-turn eval against real LLMs.

Runs the shipped teaching_turn_quality set through the eval runner with
two real transports: one drives the teaching turn (model under test), the
other judges it (a different model, so a model never grades its own
output). By default DeepSeek is under test and Claude/claude.ai judges,
--swap flips the roles.

The unit tests proved the driver, judge, and aggregation mechanisms
against fakes. Only a real run shows whether a real teacher emits a
clean teaching turn from the no-tools static intro and whether a
real judge follows the quantized-score format.

Following the smoke convention, this reports evidence rather than asserting
pass/fail: per item it prints the outcome, the per-run scores, and the
detail string. An LLM that ignores the format (a judge emitting an
unquantized score, a teacher emitting a tool call) surfaces as ERROR
outcomes with the cause visible in the detail, which is the signal to take
back to the prompt, not a test failure to silence.

Run from backend/ with:

    uv run python scripts/smoke_teaching_eval.py
    uv run python scripts/smoke_teaching_eval.py --swap
    uv run python scripts/smoke_teaching_eval.py --n-runs=3

Requires both transports configured:
  - DEEPSEEK_API_KEY in .env or environment.
  - A persistent Chrome profile logged in to claude.ai.

This run makes no database writes: the teaching eval touches no DB.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.core.config import get_settings
from app.eval.loader import load_set
from app.eval.runner import TeachingRunContext, run_set
from app.eval.schemas import TeachingEvalSet
from app.transport import TransportError
from app.transport.deepseek_impl import DeepseekTransport
from app.transport.playwright_impl import PlaywrightClaudeTransport

if TYPE_CHECKING:
    from app.eval.schemas import ItemScore
    from app.transport.base import LLMTransport

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_SET_PATH = _BACKEND_ROOT / "eval" / "sets" / "teaching_turn_quality.json"


def _print_record(record_scores: list[ItemScore], set_name: str) -> None:
    """Print each item's outcome, per-run scores, and detail as evidence."""
    print(f"  set: {set_name}")
    for score in record_scores:
        print(f"  - {score.item_id}: {score.outcome.value}")
        print(f"      scores: {score.scores}")
        print(f"      detail: {score.detail}")


async def run(*, swap: bool, n_runs: int) -> None:
    """Run the teaching set with DeepSeek and Playwright paired as teacher/judge."""
    settings = get_settings()

    loaded = load_set(_SET_PATH)
    if not isinstance(loaded, TeachingEvalSet):
        raise RuntimeError(f"Expected a teaching set, got {loaded.eval_kind.value!r}.")

    print(f"Loaded teaching set with {len(loaded.items)} items.")
    print(
        f"Roles: {'Claude teaches, DeepSeek judges' if swap else 'DeepSeek teaches, Claude judges'}"
    )
    print(f"N runs per item: {n_runs}\n")

    async with (
        DeepseekTransport(
            api_key=settings.deepseek_api_key.get_secret_value(),
            default_model=settings.deepseek_model,
        ) as ds,
        PlaywrightClaudeTransport(settings.chrome_profile_path) as pw,
    ):
        teaching_transport: LLMTransport[Any]
        judge_transport: LLMTransport[Any]
        if swap:
            teaching_transport, judge_transport = pw, ds
            transport_name = "claude_playwright"
            model_under_test = "claude.ai"
            judge_model = settings.deepseek_model
        else:
            teaching_transport, judge_transport = ds, pw
            transport_name = "deepseek"
            model_under_test = settings.deepseek_model
            judge_model = "claude.ai"

        context = TeachingRunContext(
            teaching_transport=teaching_transport,
            judge_transport=judge_transport,
            transport=transport_name,
            model_under_test=model_under_test,
            judge_model=judge_model,
            n_runs=n_runs,
        )

        record = await run_set(loaded, "smoke-not-hashed", teaching_context=context)

    _print_record(record.scores, record.set_name)

    error_count = sum(1 for s in record.scores if s.outcome.value == "error")
    print(f"\n  {len(record.scores)} items scored, {error_count} ERROR.")
    if error_count:
        print("  ERROR outcomes usually mean an LLM did not follow the format.")
        print("  Check the detail lines above for the cause (quorum, variance, parse).")
    print("\nSmoke complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="M12.4 teaching-eval smoke test.")
    parser.add_argument(
        "--swap",
        action="store_true",
        help="Swap roles: Claude teaches, DeepSeek judges (default: DeepSeek teaches).",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=3,
        help="Judged runs per item (default: 3, lower than prod default for a quick smoke).",
    )
    args = parser.parse_args()

    try:
        asyncio.run(run(swap=args.swap, n_runs=args.n_runs))
    except TransportError as e:
        print(f"\nTransport error: {e.message}")
        if e.cause is not None:
            print(f"Caused by: {type(e.cause).__name__}: {e.cause}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
