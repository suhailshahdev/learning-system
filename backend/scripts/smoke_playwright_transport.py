"""Smoke test for the Playwright + claude.ai transport.

Spins up the transport, starts a chat, sends one message, prints the
response, closes. Confirms the production transport API works end to
end against real claude.ai. Run before relying on the implementation.

Run from backend/ with:
    uv run python scripts/smoke_playwright_transport.py

Requires the persistent Chrome profile to already be logged in. Run
scripts/spike_claude_dom.py first to authenticate if needed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.transport import TransportError
from app.transport.playwright_impl import PlaywrightClaudeTransport

PROFILE_PATH = Path.home() / ".config" / "learning-system" / "chrome-profile"
TEST_PREAMBLE = "You are a test assistant. Reply briefly."
TEST_MESSAGE = "Reply with just the word 'acknowledged' and nothing else, please."


async def run_smoke() -> None:
    """Exercise the full transport lifecycle once."""
    print("Starting transport...")
    async with PlaywrightClaudeTransport(PROFILE_PATH) as transport:
        print("Transport started. Opening new chat...")
        chat = await transport.start_new_chat(TEST_PREAMBLE)
        print(f"Chat started. URL: {chat.chat_url}")
        print(f"Message count after preamble: {chat.message_count}")

        print("\nSending test message...")
        response = await transport.send(chat, TEST_MESSAGE)
        print(f"Response: {response.text!r}")
        print(f"Final URL: {chat.chat_url}")
        print(f"Final message count: {chat.message_count}")

        print("\nClosing chat...")
        await transport.close(chat)
        print("Chat closed.")
    print("\nSmoke test complete.")


def main() -> None:
    try:
        asyncio.run(run_smoke())
    except TransportError as e:
        print(f"\nTransport error: {e.message}")
        if e.cause is not None:
            print(f"Caused by: {type(e.cause).__name__}: {e.cause}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
