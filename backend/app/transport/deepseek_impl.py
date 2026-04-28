"""DeepSeek chat completions API transport implementation.

Stateless HTTP transport for the DeepSeek chat completions API.
Each send posts the full message history and gets back the next
assistant reply. The handle holds the history locally since there
is no server-side chat to manage.

The endpoint is OpenAI-compatible. We use httpx directly instead
of the OpenAI SDK since there is only one endpoint with a fixed
request shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Self

import httpx

from app.transport.base import ChatResumeMetadata, TransportError, TransportResponse

if TYPE_CHECKING:
    from types import TracebackType

    from app.transport.base import LLMTransport


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
CHAT_COMPLETIONS_PATH = "/chat/completions"

CONNECT_TIMEOUT_S = 10.0
READ_TIMEOUT_S = 120.0
TOTAL_TIMEOUT_S = 120.0


Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    """One message in a DeepSeek chat history.

    Frozen so handles can be safely shared across awaits without
    accidental mutation. The `role` literal matches the API's
    expected values; mypy catches typos at the call site.
    """

    role: Role
    content: str

    def to_wire(self) -> dict[str, str]:
        """Serialize to the JSON shape the API expects."""
        return {"role": self.role, "content": self.content}


@dataclass
class DeepseekChatHandle:
    """Per-chat state for the DeepSeek transport.

    Holds the running message history and the model name for this
    chat. message_count mirrors the same field on the Playwright
    handle so the session engine can read it the same way on either
    transport.
    """

    model: str
    history: list[Message] = field(default_factory=list)
    message_count: int = 0


class DeepseekTransport:
    """Chat completions API transport for DeepSeek.

    Owns one long-lived HTTP client for the transport's lifetime so
    connections are reused. Use as an async context manager or call
    start() and shutdown() explicitly.
    """

    def __init__(self, api_key: str, default_model: str = "deepseek-v4-flash") -> None:
        self._api_key = api_key
        self._default_model = default_model
        self._client: httpx.AsyncClient | None = None

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
        """Open the HTTP client used for every request."""
        timeout = httpx.Timeout(
            timeout=TOTAL_TIMEOUT_S,
            connect=CONNECT_TIMEOUT_S,
            read=READ_TIMEOUT_S,
        )
        self._client = httpx.AsyncClient(
            base_url=DEEPSEEK_BASE_URL,
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

    async def shutdown(self) -> None:
        """Close the HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def start_new_chat(self, system_intro: str) -> DeepseekChatHandle:
        if self._client is None:
            raise TransportError("Transport not started. Call start() first.")

        handle = DeepseekChatHandle(model=self._default_model)
        handle.history.append(Message(role="system", content=system_intro))

        # Sending an opening turn keeps both transports symmetric: after
        # start_new_chat returns, the model has acknowledged the intro
        # in both. Without this, the DeepSeek handle would have one fewer
        # round-trip than the Playwright one at the same point in the flow.
        await self._send_and_capture(handle, "Acknowledge the instructions above with one word.")
        return handle

    async def resume_chat(self, metadata: ChatResumeMetadata) -> DeepseekChatHandle:
        if self._client is None:
            raise TransportError("Transport not started. Call start() first.")

        if not metadata.prior_messages:
            raise TransportError("Cannot resume DeepSeek chat with empty prior_messages.")

        handle = DeepseekChatHandle(model=self._default_model)
        handle.history = [Message(role=m.role, content=m.content) for m in metadata.prior_messages]
        handle.message_count = metadata.message_count
        return handle

    async def send(self, chat: DeepseekChatHandle, message: str) -> TransportResponse:
        if self._client is None:
            raise TransportError("Transport not started. Call start() first.")
        return await self._send_and_capture(chat, message)

    async def close(self, chat: DeepseekChatHandle) -> None:
        # Stateless API with no per-chat resources. This method exists only
        # to satisfy the protocol uniformly; the handle is cleaned up by the
        # caller.
        return None

    async def _send_and_capture(self, chat: DeepseekChatHandle, message: str) -> TransportResponse:
        """Post one user turn, append the assistant reply to history.

        Mutates the handle's history with both the new user message
        and the assistant response on success. On failure the history
        is left as it was so the caller can retry.
        """
        if self._client is None:
            raise TransportError("Transport not started.")

        next_history = [*chat.history, Message(role="user", content=message)]
        payload = {
            "model": chat.model,
            "messages": [m.to_wire() for m in next_history],
            "stream": False,
        }

        try:
            response = await self._client.post(CHAT_COMPLETIONS_PATH, json=payload)
            response.raise_for_status()
            data: dict[str, Any] = response.json()
        except httpx.TimeoutException as e:
            raise TransportError("DeepSeek request timed out.", cause=e) from e
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            body = e.response.text[:200]
            raise TransportError(f"DeepSeek HTTP {status}: {body}", cause=e) from e
        except httpx.RequestError as e:
            raise TransportError("Network error reaching DeepSeek.", cause=e) from e
        except ValueError as e:
            raise TransportError("DeepSeek returned malformed JSON.", cause=e) from e

        choices = data.get("choices") or []
        if not choices:
            raise TransportError("DeepSeek returned no choices.")

        try:
            assistant_text = choices[0]["message"]["content"]
        except (KeyError, TypeError) as e:
            raise TransportError(
                "DeepSeek response missing choices[0].message.content.", cause=e
            ) from e

        if not isinstance(assistant_text, str):
            raise TransportError(
                f"DeepSeek returned non-string content: {type(assistant_text).__name__}."
            )

        chat.history = [*next_history, Message(role="assistant", content=assistant_text)]
        chat.message_count += 1
        return TransportResponse(text=assistant_text)


_: type[LLMTransport[DeepseekChatHandle]] = DeepseekTransport
