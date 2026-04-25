"""DeepSeek chat completions API transport implementation.

Stateless HTTP transport for DeepSeek. Each send posts the full
message history and gets back the next assistant reply. The chat
handle holds the history locally since there is no server-side
chat to manage.

Requires DEEPSEEK_API_KEY in settings. The model name is set per
handle so different chats can use different DeepSeek models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.transport.base import LLMTransport, TransportResponse


@dataclass
class DeepseekChatHandle:
    """Per-chat state for the DeepSeek transport.

    Holds the running message history (each entry is a dict with
    `role` and `content` keys, matching the API's expected shape) and
    the model name to use for this chat.
    """

    model: str
    history: list[dict[str, Any]] = field(default_factory=list)


class DeepseekTransport:
    """Chat completions API transport for DeepSeek. Implements LLMTransport[DeepseekChatHandle]."""

    async def start_new_chat(self, system_preamble: str) -> DeepseekChatHandle:
        raise NotImplementedError

    async def send(self, chat: DeepseekChatHandle, message: str) -> TransportResponse:
        raise NotImplementedError

    async def close(self, chat: DeepseekChatHandle) -> None:
        raise NotImplementedError


_: type[LLMTransport[DeepseekChatHandle]] = DeepseekTransport
