"""Protocol and shared types for LLM transports.

LLMTransport is structurally typed and parameterized over a
per-transport handle type. Each transport defines its own handle,
a Page reference for Playwright or a message history list for
DeepSeek, and the protocol just passes it through. Service code
holds the handle without knowing what is inside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeVar, runtime_checkable

Handle = TypeVar("Handle")


PriorRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class TransportResponse:
    """One response from an LLM transport.

    Single field today; structured so future fields (token counts,
    rate-limit info, model used) can be added without breaking callers.
    """

    text: str


@dataclass(frozen=True)
class PriorMessage:
    """One message from a previous turn, used when resuming a chat.

    Transports that rebuild conversation state from scratch (DeepSeek,
    where each request carries the full history) consume this. Transports
    that point at server-side state (Playwright, where claude.ai holds
    the chat) ignore it and use chat_url instead.
    """

    role: PriorRole
    content: str


@dataclass(frozen=True)
class ChatResumeMetadata:
    """Everything a transport needs to reattach to an in-progress chat.

    chat_url is set by transports that have a server-side chat to
    navigate to. prior_messages is set for transports that rebuild
    history from persisted turns. Different transports use different
    fields; building one struct with both keeps the service layer
    transport-agnostic.
    """

    chat_url: str | None = None
    prior_messages: list[PriorMessage] = field(default_factory=list)
    message_count: int = 0


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
    represented by a handle returned from start_new_chat or resume_chat
    and passed through send and close.
    """

    async def start_new_chat(self, system_intro: str) -> Handle:
        """Open a fresh chat seeded with the given system intro."""
        ...

    async def resume_chat(self, metadata: ChatResumeMetadata) -> Handle:
        """Reattach to an in-progress chat from persisted metadata."""
        ...

    async def send(self, chat: Handle, message: str) -> TransportResponse:
        """Send a user message and return the assistant's response."""
        ...

    async def close(self, chat: Handle) -> None:
        """Release any resources held by the chat handle."""
        ...
