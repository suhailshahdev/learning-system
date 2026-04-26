"""Cross-transport smoke test.

Exercises the LLMTransport contract against both implementations
through the abstract Protocol type. Confirms the abstraction holds
end to end and gives an empirical answer to whether the same
prompts work across both transports (the open question for M4's
prompt layer).

Two prompts per transport:
  1. A trivial round-trip ('reply acknowledged').
  2. A delimited-format prompt previewing M4's output shape.

Run from backend/ with:

    uv run python scripts/smoke_cross_transport.py

Requires:
  - Persistent Chrome profile logged in to claude.ai
    (run scripts/spike_claude_dom.py once if needed).
  - DEEPSEEK_API_KEY in .env or process environment.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from app.core.config import get_settings
from app.transport import TransportError
from app.transport.deepseek_impl import DeepseekTransport
from app.transport.playwright_impl import PlaywrightClaudeTransport

if TYPE_CHECKING:
    from app.transport.base import LLMTransport

TEST_PREAMBLE = "You are a test assistant. Reply briefly and follow instructions exactly."

ROUND_TRIP_PROMPT = "Reply with just the word 'acknowledged' and nothing else, please."

DELIMITED_PROMPT = """\
Reply in EXACTLY this format. Use the literal delimiters shown,
one field per block, nothing before or after the END marker.

---TOPIC---
Python > Data Types > Integers
---DIFFICULTY---
beginner
---QUESTION---
What is the result of 7 // 2 in Python 3?
---EXPECTED_ANSWER---
3
---END---
"""

REQUIRED_DELIMITERS = (
    "---TOPIC---",
    "---DIFFICULTY---",
    "---QUESTION---",
    "---EXPECTED_ANSWER---",
    "---END---",
)


async def run_round_trip(name: str, transport: LLMTransport[Any]) -> None:
    """Test 1: trivial single round-trip."""
    print(f"  [{name}] round-trip: opening chat...")
    chat = await transport.start_new_chat(TEST_PREAMBLE)

    print(f"  [{name}] round-trip: sending prompt...")
    response = await transport.send(chat, ROUND_TRIP_PROMPT)

    if not response.text.strip():
        raise RuntimeError(f"[{name}] empty response on round-trip")

    print(f"  [{name}] round-trip: response={response.text.strip()[:80]!r}")
    await transport.close(chat)
    print(f"  [{name}] round-trip: passed.\n")


async def run_delimited(name: str, transport: LLMTransport[Any]) -> None:
    """Test 2: delimited-format prompt; check structural markers.

    Answers D93 empirically: same prompt, two transports, do both
    produce parseable delimited output? Failures here change M4's
    plan; passes confirm a single prompt set is viable.
    """
    print(f"  [{name}] delimited: opening chat...")
    chat = await transport.start_new_chat(TEST_PREAMBLE)

    print(f"  [{name}] delimited: sending delimited prompt...")
    response = await transport.send(chat, DELIMITED_PROMPT)

    text = response.text
    missing = [d for d in REQUIRED_DELIMITERS if d not in text]
    if missing:
        print(f"  [{name}] delimited: response was:\n---\n{text}\n---")
        raise RuntimeError(f"[{name}] response missing delimiters: {missing}")

    # Order check: delimiters should appear in the canonical sequence.
    positions = [text.index(d) for d in REQUIRED_DELIMITERS]
    if positions != sorted(positions):
        raise RuntimeError(f"[{name}] delimiters out of order. Positions: {positions}")

    print(f"  [{name}] delimited: all delimiters present and ordered.")
    await transport.close(chat)
    print(f"  [{name}] delimited: passed.\n")


async def exercise_transport(name: str, transport: LLMTransport[Any]) -> None:
    """Run both tests against one transport."""
    print(f"=== {name} ===")
    await run_round_trip(name, transport)
    await run_delimited(name, transport)


async def run_smoke() -> None:
    """Exercise both transports through the LLMTransport Protocol."""
    settings = get_settings()

    # Playwright first because it has the heavier startup. If the
    # profile is logged out we want to know before paying for any
    # DeepSeek calls.
    print("Starting Playwright transport...\n")
    async with PlaywrightClaudeTransport(settings.chrome_profile_path) as pw:
        await exercise_transport("Playwright/claude.ai", pw)

    print("Starting DeepSeek transport...\n")
    async with DeepseekTransport(
        api_key=settings.deepseek_api_key.get_secret_value(),
        default_model=settings.deepseek_model,
    ) as ds:
        await exercise_transport(f"DeepSeek/{settings.deepseek_model}", ds)

    print("Cross-transport smoke complete. Both transports honor the LLMTransport contract.")


def main() -> None:
    try:
        asyncio.run(run_smoke())
    except TransportError as e:
        print(f"\nTransport error: {e.message}")
        if e.cause is not None:
            print(f"Caused by: {type(e.cause).__name__}: {e.cause}")
        raise SystemExit(1) from e
    except RuntimeError as e:
        print(f"\nAssertion failed: {e}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
