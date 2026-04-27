"""Protocol and shared types for LLM transports.

LLMTransport is structurally typed and parameterized over a
per-transport handle type. Each transport defines its own handle,
a Page reference for Playwright or a message history list for
DeepSeek, and the protocol just passes it through. Service code
holds the handle without knowing what is inside it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

Handle = TypeVar("Handle")


@dataclass(frozen=True)
class TransportResponse:
    """One response from an LLM transport.

    Single field today; structured so future fields (token counts,
    rate-limit info, model used) can be added without breaking callers.
    """

    text: str


class TransportError(Exception):
    """An LLM transport operation failed.

    Carries a human-readable message and an optional underlying cause.
    Service code catches this, logs to `error_log`, and surfaces a
    clear message to the user without losing session progress.
    """

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause


@runtime_checkable
class LLMTransport(Protocol[Handle]):
    """Common interface for every LLM transport.

    A transport manages one or more chats with an LLM. Each chat is
    represented by a handle returned from start_new_chat and passed
    through send and close.
    """

    async def start_new_chat(self, system_intro: str) -> Handle:
        """Open a fresh chat seeded with the given system intro."""
        ...

    async def send(self, chat: Handle, message: str) -> TransportResponse:
        """Send a user message and return the assistant's response."""
        ...

    async def close(self, chat: Handle) -> None:
        """Release any resources held by the chat handle."""
        ...
