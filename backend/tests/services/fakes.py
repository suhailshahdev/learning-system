"""Test fakes for service-layer tests.

A FakeTransport satisfies the LLMTransport Protocol with canned
responses. Used by service tests so they exercise real service
logic without hitting any LLM.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.transport.base import ToolResult, TransportError, TransportResponse

if TYPE_CHECKING:
    from app.transport.base import ChatResumeMetadata, LLMTransport


@dataclass
class FakeChatHandle:
    """Per-chat state for the fake transport.

    Holds the messages this chat has seen so tests can assert on
    what the service sent. message_count mirrors the real handles.
    resumed_from is set when the handle came from resume_chat so
    tests can assert which entry point produced it.

    tool_results_received accumulates the lists of ToolResult sent
    via send_tool_results so tests can assert on tool execution.
    """

    messages_sent: list[str] = field(default_factory=list)
    message_count: int = 0
    chat_url: str | None = None
    resumed_from: ChatResumeMetadata | None = None
    tool_results_received: list[list[ToolResult]] = field(default_factory=list)


class FakeTransport:
    """In-memory transport returning canned responses in order.

    Constructor takes the list of responses to return on each
    successive send. Each response can be a plain string (treated
    as TransportResponse(text=...)) or a TransportResponse for
    tests that need to surface tool_calls. Tests assert on what
    was sent (via handle.messages_sent and handle.tool_results_received)
    and on what the service did with the response (via DB rows or
    the parser's output).

    Set raise_on_send to a TransportError instance to simulate a
    transport failure on the next send call.
    """

    def __init__(
        self,
        responses: list[str | TransportResponse],
        *,
        raise_on_send: TransportError | None = None,
        raise_on_send_at: int | None = None,
    ) -> None:
        self._responses: deque[TransportResponse] = deque(
            r if isinstance(r, TransportResponse) else TransportResponse(text=r) for r in responses
        )
        self._raise_on_send = raise_on_send
        self._raise_on_send_at = raise_on_send_at
        self._send_call_count = 0
        self.chats: list[FakeChatHandle] = []

    async def start_new_chat(
        self, system_intro: str, first_message: str
    ) -> tuple[FakeChatHandle, TransportResponse]:
        # When raise_on_send_at is set, the test wants a specific send()
        # call to fail and start_new_chat is expected to succeed. When
        # raise_on_send_at is None, the legacy behavior holds and any
        # transport call raises.
        if self._raise_on_send is not None and self._raise_on_send_at is None:
            raise self._raise_on_send

        if not self._responses:
            raise RuntimeError(
                "FakeTransport exhausted: start_new_chat() called but no canned responses left."
            )
        chat = FakeChatHandle()
        chat.messages_sent.append(system_intro)
        chat.messages_sent.append(first_message)
        chat.message_count += 1
        self.chats.append(chat)
        return chat, self._responses.popleft()

    async def resume_chat(self, metadata: ChatResumeMetadata) -> FakeChatHandle:
        chat = FakeChatHandle(
            message_count=metadata.message_count,
            chat_url=metadata.chat_url,
            resumed_from=metadata,
        )
        self.chats.append(chat)
        return chat

    async def send(self, chat: FakeChatHandle, message: str) -> TransportResponse:
        if self._raise_on_send is not None and (
            self._raise_on_send_at is None or self._raise_on_send_at == self._send_call_count
        ):
            self._send_call_count += 1
            raise self._raise_on_send

        self._send_call_count += 1
        chat.messages_sent.append(message)
        chat.message_count += 1
        if not self._responses:
            raise RuntimeError(
                "FakeTransport exhausted: send() called but no canned responses left."
            )
        return self._responses.popleft()

    async def send_tool_results(
        self, chat: FakeChatHandle, results: list[ToolResult]
    ) -> TransportResponse:
        """Record the tool results and return the next canned response.

        Symmetric with real transports: tests can assert what tool
        results the service sent by reading chat.tool_results_received,
        and the next canned response simulates the LLM's reply after
        seeing those results.
        """
        chat.tool_results_received.append(results)
        chat.message_count += 1
        if not self._responses:
            raise RuntimeError(
                "FakeTransport exhausted: send_tool_results() called but no canned responses left."
            )
        return self._responses.popleft()

    async def close(self, chat: FakeChatHandle) -> None:
        return None


_: type[LLMTransport[FakeChatHandle]] = FakeTransport
