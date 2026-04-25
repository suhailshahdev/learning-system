"""Playwright + claude.ai transport implementation.

Drives a persistent Chrome profile via Patchright to bypass
Cloudflare's bot detection on claude.ai. The persistent profile
keeps the user logged in across runs after the first login.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.transport.base import LLMTransport, TransportResponse


@dataclass
class PlaywrightChatHandle:
    """Per-chat state for the Playwright + claude.ai transport.

    Holds the chat URL once one is created, the running message count
    we track ourselves (claude.ai surfaces no in-UI counter), and any
    Playwright-side references the transport needs.
    """

    chat_url: str | None = None
    message_count: int = 0


class PlaywrightClaudeTransport:
    """Browser-automation transport for claude.ai. Implements LLMTransport[PlaywrightChatHandle]."""

    async def start_new_chat(self, system_preamble: str) -> PlaywrightChatHandle:
        raise NotImplementedError

    async def send(self, chat: PlaywrightChatHandle, message: str) -> TransportResponse:
        raise NotImplementedError

    async def close(self, chat: PlaywrightChatHandle) -> None:
        raise NotImplementedError


_: type[LLMTransport[PlaywrightChatHandle]] = PlaywrightClaudeTransport
