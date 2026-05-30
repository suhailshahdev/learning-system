"""Observability wrapper around any LLM transport.

InstrumentedTransport wraps a concrete transport and records one
llm_call row per round-trip. It is structurally an LLMTransport
itself, so service code holds it without knowing it is wrapped.

The wrapper sees only the Protocol surface: a system intro, a
message, a handle, a response. It cannot see which transport it
wraps or what model that transport uses, so both are passed at
construction. The lifespan knows which transport it is wrapping
and supplies them.

Four of the five Protocol methods produce or consume an LLM
round-trip and are recorded: start_new_chat, send,
send_tool_results, and resume_chat. resume_chat does no LLM call
on either transport today, but a cold resume can still fail
loading chat history, and that failure should not be invisible.
close holds no round-trip and passes straight through.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from uuid import uuid4

from app.services.llm_call_recorder import LLMCallData
from app.transport.base import TransportError, TransportResponse

if TYPE_CHECKING:
    from app.models import TransportKind
    from app.services.llm_call_recorder import LLMCallRecorder
    from app.transport.base import (
        ChatResumeMetadata,
        LLMTransport,
        ToolResult,
    )


class InstrumentedTransport[Handle]:
    """Wraps a transport and records one llm_call per round-trip.

    Generic over the wrapped transport's handle so the handle
    passes through unchanged. transport_kind and model are fixed
    for the wrapped transport's lifetime and supplied at
    construction because the Protocol does not expose them.

    Structurally an LLMTransport, the assertion at the bottom of
    this module proves conformance the same way the concrete
    transports do, without subclassing the Protocol (which would
    force LLMTransport and Handle to runtime imports).
    """

    def __init__(
        self,
        inner: LLMTransport[Handle],
        recorder: LLMCallRecorder,
        transport_kind: TransportKind,
        model: str | None = None,
    ) -> None:
        self._inner = inner
        self._recorder = recorder
        self._transport_kind = transport_kind
        self._model = model

    async def start_new_chat(
        self, system_intro: str, first_message: str
    ) -> tuple[Handle, TransportResponse]:
        prompt_chars = len(system_intro) + len(first_message)
        start = time.perf_counter()
        try:
            handle, response = await self._inner.start_new_chat(system_intro, first_message)
        except TransportError as e:
            self._record("start_new_chat", start, prompt_chars, None, e)
            raise
        self._record("start_new_chat", start, prompt_chars, response, None)
        return handle, response

    async def resume_chat(self, metadata: ChatResumeMetadata) -> Handle:
        start = time.perf_counter()
        try:
            handle = await self._inner.resume_chat(metadata)
        except TransportError as e:
            self._record("resume_chat", start, 0, None, e)
            raise
        self._record("resume_chat", start, 0, None, None)
        return handle

    async def send(self, chat: Handle, message: str) -> TransportResponse:
        start = time.perf_counter()
        try:
            response = await self._inner.send(chat, message)
        except TransportError as e:
            self._record("send", start, len(message), None, e)
            raise
        self._record("send", start, len(message), response, None)
        return response

    async def send_tool_results(self, chat: Handle, results: list[ToolResult]) -> TransportResponse:
        prompt_chars = sum(len(r.content) for r in results)
        start = time.perf_counter()
        try:
            response = await self._inner.send_tool_results(chat, results)
        except TransportError as e:
            self._record("send_tool_results", start, prompt_chars, None, e)
            raise
        self._record("send_tool_results", start, prompt_chars, response, None)
        return response

    async def close(self, chat: Handle) -> None:
        await self._inner.close(chat)

    def _record(
        self,
        method: str,
        start: float,
        prompt_chars: int,
        response: TransportResponse | None,
        error: TransportError | None,
    ) -> None:
        """Build one LLMCallData and hand it to the recorder.

        Called on both the success and failure path of every
        recorded method. response is None on failure or when the
        method returns no TransportResponse (resume_chat). The
        recorder never raises, so this needs no guard of its own.
        """
        latency_ms = int((time.perf_counter() - start) * 1000)
        self._recorder.record(
            LLMCallData(
                trace_id=str(uuid4()),
                transport_kind=self._transport_kind,
                method=method,
                latency_ms=latency_ms,
                prompt_chars=prompt_chars,
                response_chars=len(response.text) if response is not None else 0,
                success=error is None,
                model=self._model,
                error=error.message if error is not None else None,
            )
        )


# Structural conformance check, same pattern as the concrete
# transports. If a Protocol method signature drifts, mypy fails here.
if TYPE_CHECKING:
    _: type[LLMTransport[object]] = InstrumentedTransport[object]


__all__ = ["InstrumentedTransport"]
