"""Playwright + claude.ai transport implementation.

Drives a persistent Chrome profile via Patchright to bypass
Cloudflare's bot detection on claude.ai. The persistent profile
keeps the user logged in across runs after the first login.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

from patchright.async_api import (
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
)
from patchright.async_api import (
    TimeoutError as PlaywrightTimeout,
)

from app.transport.base import (
    ChatResumeMetadata,
    ToolResult,
    TransportError,
    TransportResponse,
)

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType

    from app.transport.base import LLMTransport


CLAUDE_BASE_URL = "https://claude.ai"
CLAUDE_NEW_CHAT_URL = "https://claude.ai/new"

INPUT_SELECTOR = 'div[contenteditable="true"]'
ASSISTANT_MESSAGE_SELECTOR = "div.font-claude-response"
STREAMING_DONE_ATTR = "data-is-streaming"
STREAMING_DONE_VALUE = "false"

PAGE_LOAD_TIMEOUT_MS = 30_000
INPUT_READY_TIMEOUT_MS = 10_000
HISTORY_LOAD_TIMEOUT_MS = 15_000
RESPONSE_START_TIMEOUT_MS = 30_000
RESPONSE_DONE_TIMEOUT_MS = 5 * 60_000


@dataclass
class PlaywrightChatHandle:
    """Per-chat state for the Playwright + claude.ai transport.

    Holds the Page driving this chat, the chat URL once claude.ai
    assigns one (after the first message), and the running message
    count tracked here because claude.ai surfaces no in-UI counter.
    """

    page: Page
    chat_url: str | None = None
    message_count: int = 0


class PlaywrightClaudeTransport:
    """Browser-automation transport for claude.ai.

    Holds one persistent browser context for the transport's lifetime;
    each chat gets its own page within that context. Use as an async
    context manager, or call `start()` and `shutdown()` explicitly.
    """

    def __init__(self, chrome_profile_path: Path) -> None:
        self._profile_path = chrome_profile_path
        self._playwright: Playwright | None = None
        self._context: BrowserContext | None = None
        # claude.ai treats two Pages on the same /chat/<uuid> URL as a
        # conflict: one renders the assistant messages, the other does not.
        # The transport keeps Pages alive across calls and reuses them when
        # resume_chat targets a URL it already owns. Bug verified by the
        # debug_resume_dom script
        self._pages_by_url: dict[str, Page] = {}

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.shutdown()

    async def start(self) -> None:
        """Launch the browser and verify the profile is logged in."""
        self._profile_path.mkdir(parents=True, exist_ok=True)
        self._playwright = await async_playwright().start()
        try:
            self._context = await self._playwright.chromium.launch_persistent_context(
                user_data_dir=str(self._profile_path),
                channel="chrome",
                headless=False,
                no_viewport=True,
            )
        except Exception as e:
            await self._playwright.stop()
            self._playwright = None
            raise TransportError("Failed to launch browser context.", cause=e) from e

        await self._verify_logged_in()

    async def shutdown(self) -> None:
        """Close the browser context and stop Playwright."""
        self._pages_by_url.clear()
        if self._context is not None:
            await self._context.close()
            self._context = None
        if self._playwright is not None:
            await self._playwright.stop()
            self._playwright = None

    async def start_new_chat(
        self, system_intro: str, first_message: str
    ) -> tuple[PlaywrightChatHandle, TransportResponse]:
        if self._context is None:
            raise TransportError("Transport not started. Call start() first.")

        page = await self._context.new_page()
        try:
            await page.goto(CLAUDE_NEW_CHAT_URL, timeout=PAGE_LOAD_TIMEOUT_MS)
            await page.wait_for_selector(INPUT_SELECTOR, timeout=INPUT_READY_TIMEOUT_MS)
        except PlaywrightTimeout as e:
            await page.close()
            raise TransportError(
                "Could not reach claude.ai/new or input never appeared.", cause=e
            ) from e

        handle = PlaywrightChatHandle(page=page)
        combined = f"{system_intro}\n\n{first_message}"
        response = await self._send_and_capture(handle, combined)

        # The chat URL is assigned by claude.ai after the first message.
        # Register the Page now so resume_chat can reuse it later.
        if handle.chat_url is not None:
            self._pages_by_url[handle.chat_url] = page

        return handle, response

    async def resume_chat(self, metadata: ChatResumeMetadata) -> PlaywrightChatHandle:
        if self._context is None:
            raise TransportError("Transport not started. Call start() first.")

        if metadata.chat_url is None:
            raise TransportError("Cannot resume Playwright chat without chat_url.")

        # Warm path: a Page is already open on this URL from start_new_chat
        # earlier in the same process. Reuse it — opening a second Page on
        # the same URL leaves the second one with an empty assistant slot.
        existing_page = self._pages_by_url.get(metadata.chat_url)
        if existing_page is not None and not existing_page.is_closed():
            return PlaywrightChatHandle(
                page=existing_page,
                chat_url=metadata.chat_url,
                message_count=metadata.message_count,
            )

        # Cold path: no Page in memory (process restart, or close was
        # called). Open a fresh Page on the URL and verify history loaded.
        page = await self._context.new_page()
        try:
            await page.goto(metadata.chat_url, timeout=PAGE_LOAD_TIMEOUT_MS)
            await page.wait_for_selector(INPUT_SELECTOR, timeout=INPUT_READY_TIMEOUT_MS)
            await page.wait_for_selector(
                ASSISTANT_MESSAGE_SELECTOR, timeout=HISTORY_LOAD_TIMEOUT_MS
            )
        except PlaywrightTimeout as e:
            await page.close()
            raise TransportError(
                f"Could not load chat history at {metadata.chat_url}. "
                "The chat may have been deleted, or claude.ai's DOM may have changed.",
                cause=e,
            ) from e

        self._pages_by_url[metadata.chat_url] = page
        return PlaywrightChatHandle(
            page=page,
            chat_url=metadata.chat_url,
            message_count=metadata.message_count,
        )

    async def send(self, chat: PlaywrightChatHandle, message: str) -> TransportResponse:
        if chat.page.is_closed():
            raise TransportError("Chat page has been closed externally.")
        return await self._send_and_capture(chat, message)

    async def close(self, chat: PlaywrightChatHandle) -> None:
        if chat.chat_url is not None:
            self._pages_by_url.pop(chat.chat_url, None)
        if not chat.page.is_closed():
            await chat.page.close()

    async def send_tool_results(
        self, chat: PlaywrightChatHandle, results: list[ToolResult]
    ) -> TransportResponse:
        """Send tool execution results back as a delimited user message.

        claude.ai has no `tool` role channel, so results are sent as
        plain user messages wrapped in ---TOOL_RESULT--- blocks. The
        intro tells the LLM to read content blocks and continue. One
        block per result, separated by blank lines, sent in a single
        user message to keep the message-count budget tight.
        """
        if chat.page.is_closed():
            raise TransportError("Chat page has been closed externally.")

        message = self._format_tool_results(results)
        return await self._send_and_capture(chat, message)

    @staticmethod
    def _format_tool_results(results: list[ToolResult]) -> str:
        """Format tool results as a sequence of delimited blocks."""
        blocks = [
            f"---TOOL_RESULT---\n"
            f"{json.dumps({'call_id': result.call_id, 'content': result.content})}\n"
            f"---END---"
            for result in results
        ]
        return "\n\n".join(blocks)

    async def _verify_logged_in(self) -> None:
        """Open claude.ai once at startup and confirm we're authenticated."""
        if self._context is None:
            raise TransportError("Transport not started.")

        page = await self._context.new_page()
        try:
            await page.goto(CLAUDE_BASE_URL, timeout=PAGE_LOAD_TIMEOUT_MS)
            try:
                await page.wait_for_url(
                    lambda url: "/login" not in url, timeout=INPUT_READY_TIMEOUT_MS
                )
            except PlaywrightTimeout as e:
                raise TransportError(
                    "Profile is not logged in to claude.ai. "
                    "Run `uv run python scripts/spike_claude_dom.py` once "
                    "and complete the login flow, then start the backend.",
                    cause=e,
                ) from e
        finally:
            await page.close()

    async def _send_and_capture(
        self, chat: PlaywrightChatHandle, message: str
    ) -> TransportResponse:
        """Send one message and capture the assistant's response.

        Counts the assistant messages before submitting, then waits for
        a new one to appear and finish streaming. The new element is
        the response; reading the DOM count avoids the spike's bug of
        picking up the user's own message.
        """
        page = chat.page

        try:
            existing = await page.locator(ASSISTANT_MESSAGE_SELECTOR).count()

            input_box = page.locator(INPUT_SELECTOR).first
            await input_box.click()
            await input_box.fill(message)
            await page.keyboard.press("Enter")

            await page.wait_for_function(
                f"""
                () => document.querySelectorAll(
                    {ASSISTANT_MESSAGE_SELECTOR!r}
                ).length > {existing}
                """,
                timeout=RESPONSE_START_TIMEOUT_MS,
            )

            new_message = page.locator(ASSISTANT_MESSAGE_SELECTOR).nth(existing)
            await new_message.wait_for(state="attached")

            await page.wait_for_function(
                f"""
                () => {{
                    const elements = document.querySelectorAll(
                        {ASSISTANT_MESSAGE_SELECTOR!r}
                    );
                    if (elements.length <= {existing}) return false;
                    const target = elements[{existing}].closest(
                        "[{STREAMING_DONE_ATTR}]"
                    );
                    return target && target.getAttribute(
                        {STREAMING_DONE_ATTR!r}
                    ) === {STREAMING_DONE_VALUE!r};
                }}
                """,
                timeout=RESPONSE_DONE_TIMEOUT_MS,
            )

            text = await new_message.text_content() or ""
        except PlaywrightTimeout as e:
            raise TransportError("Timed out waiting for claude.ai response.", cause=e) from e
        except Exception as e:
            raise TransportError(f"Unexpected error during send: {e}", cause=e) from e

        chat.message_count += 1
        chat.chat_url = page.url
        return TransportResponse(text=text)


_: type[LLMTransport[PlaywrightChatHandle]] = PlaywrightClaudeTransport
