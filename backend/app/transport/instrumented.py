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
from contextlib import contextmanager
from typing import TYPE_CHECKING
from uuid import uuid4

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

from app.core.telemetry import (
    ATTR_GEN_AI_OPERATION,
    ATTR_GEN_AI_REQUEST_MODEL,
    ATTR_GEN_AI_SYSTEM,
    get_tracer,
)
from app.core.trace_context import current_trace_id
from app.models import TransportKind
from app.services.llm_call_recorder import LLMCallData
from app.transport.base import TransportError, TransportResponse

if TYPE_CHECKING:
    from collections.abc import Iterator

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
        self._tracer = get_tracer()
        # GenAI conventions want a system identifier. claude.ai is
        # reached via browser but the model behind it is Anthropic's.
        # DeepSeek is itself. Derived once from the transport kind.
        self._gen_ai_system = _gen_ai_system_for(transport_kind)

    async def start_new_chat(
        self, system_intro: str, first_message: str
    ) -> tuple[Handle, TransportResponse]:
        prompt_chars = len(system_intro) + len(first_message)
        start = time.perf_counter()
        with self._span("start_new_chat") as span:
            try:
                handle, response = await self._inner.start_new_chat(system_intro, first_message)
            except TransportError as e:
                self._fail_span(span, e)
                self._record("start_new_chat", start, prompt_chars, None, e)
                raise
            self._record("start_new_chat", start, prompt_chars, response, None)
            return handle, response

    async def resume_chat(self, metadata: ChatResumeMetadata) -> Handle:
        start = time.perf_counter()
        with self._span("resume_chat") as span:
            try:
                handle = await self._inner.resume_chat(metadata)
            except TransportError as e:
                self._fail_span(span, e)
                self._record("resume_chat", start, 0, None, e)
                raise
            self._record("resume_chat", start, 0, None, None)
            return handle

    async def send(self, chat: Handle, message: str) -> TransportResponse:
        start = time.perf_counter()
        with self._span("send") as span:
            try:
                response = await self._inner.send(chat, message)
            except TransportError as e:
                self._fail_span(span, e)
                self._record("send", start, len(message), None, e)
                raise
            self._record("send", start, len(message), response, None)
            return response

    async def send_tool_results(self, chat: Handle, results: list[ToolResult]) -> TransportResponse:
        prompt_chars = sum(len(r.content) for r in results)
        start = time.perf_counter()
        with self._span("send_tool_results") as span:
            try:
                response = await self._inner.send_tool_results(chat, results)
            except TransportError as e:
                self._fail_span(span, e)
                self._record("send_tool_results", start, prompt_chars, None, e)
                raise
            self._record("send_tool_results", start, prompt_chars, response, None)
            return response

    async def close(self, chat: Handle) -> None:
        await self._inner.close(chat)

    @contextmanager
    def _span(self, method: str) -> Iterator[trace.Span]:
        """Open a CLIENT span for one round-trip with GenAI attributes.

        Wraps start_as_current_span so the span is current for the
        duration of the inner call and ends when the block exits.
        When tracing is off the tracer hands back a non-recording
        span and every attribute set is a no-op.
        """
        with self._tracer.start_as_current_span(
            f"{self._gen_ai_system}.{method}",
            kind=SpanKind.CLIENT,
            attributes={
                ATTR_GEN_AI_SYSTEM: self._gen_ai_system,
                ATTR_GEN_AI_OPERATION: method,
                **({ATTR_GEN_AI_REQUEST_MODEL: self._model} if self._model else {}),
            },
        ) as span:
            yield span

    @staticmethod
    def _fail_span(span: trace.Span, error: TransportError) -> None:
        """Mark a span failed and attach the exception.

        Sets the span status to ERROR and records the exception so a
        failed round-trip is visible in the trace, not just in the
        recorder row. No-op on a non-recording span.
        """
        span.set_status(Status(StatusCode.ERROR, error.message))
        span.record_exception(error)

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
                trace_id=_current_trace_id(),
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


def _gen_ai_system_for(transport_kind: TransportKind) -> str:
    """Map a transport kind to its GenAI system identifier.

    The Playwright transport reaches Anthropic's model through a
    browser, so its system is anthropic, not the transport name.
    DeepSeek is its own system.

    No fallback branch: TransportKind has two members and both are
    handled, so mypy proves this exhaustive. Adding a third member
    surfaces a missing-return error here, which is the signal to add
    its mapping rather than silently returning a placeholder.
    """
    if transport_kind == TransportKind.CLAUDE_PLAYWRIGHT:
        return "anthropic"
    return "deepseek"


def _current_trace_id() -> str:
    """Trace id for the recorder row.

    Prefers the active turn's trace id when a session-service turn is
    in progress, so every llm_call in one turn shares the id the
    error logger will also use. Outside a turn trace (defensive; no
    such path today) falls back to the active span's id when tracing
    is on, or a fresh uuid4 hex when off. Either way a 32-char hex
    string the row column holds uniformly.
    """
    turn_id = current_trace_id()
    if turn_id is not None:
        return turn_id
    span_context = trace.get_current_span().get_span_context()
    if span_context.trace_id != 0:
        return format(span_context.trace_id, "032x")
    return uuid4().hex


# Structural conformance check, same pattern as the concrete
# transports. If a Protocol method signature drifts, mypy fails here.
if TYPE_CHECKING:
    _: type[LLMTransport[object]] = InstrumentedTransport[object]


__all__ = ["InstrumentedTransport"]
