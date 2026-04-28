"""Test fakes for service-layer tests.

A FakeTransport satisfies the LLMTransport Protocol with canned
responses. Used by service tests so they exercise real service
logic without hitting any LLM.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.transport.base import TransportError, TransportResponse

if TYPE_CHECKING:
    from app.transport.base import ChatResumeMetadata, LLMTransport


@dataclass
class FakeChatHandle:
    """Per-chat state for the fake transport.

    Holds the messages this chat has seen so tests can assert on
    what the service sent. message_count mirrors the real handles.
    resumed_from is set when the handle came from resume_chat so
    tests can assert which entry point produced it.
    """

    messages_sent: list[str] = field(default_factory=list)
    message_count: int = 0
    chat_url: str | None = None
    resumed_from: ChatResumeMetadata | None = None


class FakeTransport:
    """In-memory transport returning canned responses in order.

    Constructor takes the list of response texts to return on each
    successive send. Tests assert on what was sent (via
    handle.messages_sent) and on what the service did with the
    response (via DB rows or the parser's output).

    Set raise_on_send to a TransportError instance to simulate a
    transport failure on the next send call.
    """

    def __init__(
        self,
        responses: list[str],
        *,
        raise_on_send: TransportError | None = None,
    ) -> None:
        self._responses: deque[str] = deque(responses)
        self._raise_on_send = raise_on_send
        self.chats: list[FakeChatHandle] = []

    async def start_new_chat(self, system_intro: str) -> FakeChatHandle:
        chat = FakeChatHandle()
        chat.messages_sent.append(system_intro)
        self.chats.append(chat)
        return chat

    async def resume_chat(self, metadata: ChatResumeMetadata) -> FakeChatHandle:
        chat = FakeChatHandle(
            message_count=metadata.message_count,
            chat_url=metadata.chat_url,
            resumed_from=metadata,
        )
        self.chats.append(chat)
        return chat

    async def send(self, chat: FakeChatHandle, message: str) -> TransportResponse:
        if self._raise_on_send is not None:
            raise self._raise_on_send
        chat.messages_sent.append(message)
        chat.message_count += 1
        if not self._responses:
            raise RuntimeError(
                "FakeTransport exhausted: send() called but no canned responses left."
            )
        return TransportResponse(text=self._responses.popleft())

    async def close(self, chat: FakeChatHandle) -> None:
        return None


_: type[LLMTransport[FakeChatHandle]] = FakeTransport
