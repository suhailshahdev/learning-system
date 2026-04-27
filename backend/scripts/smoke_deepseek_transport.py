"""Smoke test for the DeepSeek chat completions transport.

Spins up the transport, starts a chat, sends one message, prints the
response, closes. Confirms the production transport works end to end
against the live DeepSeek API. Run before relying on the implementation.

Run from backend/ with:

    uv run python scripts/smoke_deepseek_transport.py

Requires DEEPSEEK_API_KEY in .env or the process environment.
"""

from __future__ import annotations

import asyncio

from app.core.config import get_settings
from app.transport import TransportError
from app.transport.deepseek_impl import DeepseekTransport

TEST_INTRO = "You are a test assistant. Reply briefly."
TEST_MESSAGE = "Reply with just the word 'acknowledged' and nothing else, please."


async def run_smoke() -> None:
    """Exercise the full transport lifecycle once."""
    settings = get_settings()

    print(f"Starting transport (model={settings.deepseek_model})...")
    async with DeepseekTransport(
        api_key=settings.deepseek_api_key.get_secret_value(),
        default_model=settings.deepseek_model,
    ) as transport:
        print("Transport started. Opening new chat...")
        chat = await transport.start_new_chat(TEST_INTRO)
        print(f"Chat started. Model: {chat.model}")
        print(f"Message count after intro: {chat.message_count}")
        print(f"History length: {len(chat.history)}")

        print("\nSending test message...")
        response = await transport.send(chat, TEST_MESSAGE)
        print(f"Response: {response.text!r}")
        print(f"Final message count: {chat.message_count}")
        print(f"Final history length: {len(chat.history)}")

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
