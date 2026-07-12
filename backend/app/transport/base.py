"""Protocol and shared types for LLM transports.

LLMTransport is structurally typed and parameterized over a
per-transport handle type. Each transport defines its own handle,
a Page reference for Playwright or a message history list for
DeepSeek, and the protocol just passes it through. Service code
holds the handle without knowing what is inside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from app.schemas.tools import ToolCall

Handle = TypeVar("Handle")
PriorRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class TransportResponse:
    """One response from an LLM transport.

    Carries either the LLM's text response or a list of tool calls
    to execute. The two are mutually exclusive in any single turn:
    the LLM either responds with text or requests tool execution.

    `text` is the LLM's response when it returned text. Empty when
    the response was tool calls only.

    `tool_calls` is a list of structured tool invocations. Empty
    when the response was plain text. Non-empty when the LLM is
    asking for tool execution. The session-service loop runs each
    via the registry and feeds results back via send_tool_results.

    Both transports surface tool calls through this same field.
    DeepSeek populates it from the API's native tool_calls response
    field. Claude transport populates it after parsing a
    ---TOOL_CALL--- block from chat text.
    """

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass(frozen=True)
class ToolResult:
    """The result of executing one tool call.

    Sent back to the LLM via send_tool_results so the LLM can see
    what the tool returned and continue. `call_id` matches the id
    the LLM provided when it requested the tool. DeepSeek's API requires
    this for correlation. The Playwright transport does not need it
    but carries the field for symmetry.

    `content` is a JSON string of the tool's output. The transport
    decides how to format this in the next API call (tool role
    message for DeepSeek, user message for Claude transport).
    """

    call_id: str
    content: str


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
    fields, building one struct with both keeps the service layer
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
    and passed through send, send_tool_results, and close.

    Responses can carry either text or tool calls. When the LLM
    requests tool execution, the session-service loop runs each
    tool via the registry and sends the results back via
    send_tool_results, then waits for the LLM's next response.
    """

    async def start_new_chat(
        self,
        system_intro: str,
        first_message: str,
        tool_names: Sequence[str] | None = None,
    ) -> tuple[Handle, TransportResponse]:
        """Open a fresh chat with the intro and send the first user message.

        Returns the handle and the assistant's response to the first
        message. Combining intro and first message into one call lets
        transports that have no native system-role channel (claude.ai)
        avoid producing a separate onboarding turn before the real
        teaching turn lands.

        tool_names is the chat's native tool surface: exactly the
        tools the transport advertises through its API for this
        chat's lifetime. None advertises nothing. Callers pass the
        set their flow allows, so the API-level offer can never
        contradict the flow's gate. Transports without a native tool
        channel (claude.ai) ignore it; their intro prose is the only
        advertisement.

        The response may carry tool_calls. The session-service loop
        handles them and calls send_tool_results.
        """
        ...

    async def resume_chat(
        self, metadata: ChatResumeMetadata, tool_names: Sequence[str] | None = None
    ) -> Handle:
        """Reattach to an in-progress chat from persisted metadata.

        tool_names re-establishes the chat's native tool surface,
        which is not part of the persisted metadata. Same semantics
        as start_new_chat.
        """
        ...

    async def send(self, chat: Handle, message: str) -> TransportResponse:
        """Send a user message and return the assistant's response.

        The response may carry tool_calls. The session-service loop
        handles them and calls send_tool_results.
        """
        ...

    async def send_tool_results(self, chat: Handle, results: list[ToolResult]) -> TransportResponse:
        """Send tool execution results back to the LLM.

        Used after a previous response carried tool_calls. The
        transport formats the results appropriately for its API
        (tool role messages for OpenAI-compatible APIs, user
        messages for chat-text-based transports) and returns the
        LLM's next response.

        The next response may itself carry tool_calls if the LLM
        wants to chain calls before producing teaching content.
        """
        ...

    async def close(self, chat: Handle) -> None:
        """Release any resources held by the chat handle."""
        ...
