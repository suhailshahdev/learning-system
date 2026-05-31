"""Per-turn trace context shared across a session-service call.

A session-service entry point (start_session, send_user_answer,
request_next_question) runs one logical turn that may make several
LLM round-trips and may write an error_log row if something fails.
Everything in that turn should carry one trace id so an error and
the LLM call that led to it can be joined.

The wrapper that records llm_call rows and the helper that writes
error_log rows are far apart in the call stack and do not pass a
trace id between them. A context variable bridges that gap: the
entry point opens a turn trace, the wrapper and the error logger
both read it, and they converge on the same id without threading
it through every signature.

This is a half-step toward full OpenTelemetry trace propagation.
When a turn-level span exists (the multi-agent work), its trace id
feeds this context and the LLM spans nest under it. For now the id
is the active span's when tracing is on, or a fresh uuid4 when off,
so the linkage works whether or not spans are being exported.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING
from uuid import uuid4

from opentelemetry import trace

if TYPE_CHECKING:
    from collections.abc import Iterator

# None when no turn trace is active. Set for the duration of a
# session-service entry point's work. Per-task by construction, so
# concurrent sessions never see each other's id.
_current_trace_id: ContextVar[str | None] = ContextVar("current_trace_id", default=None)


def _new_trace_id() -> str:
    """Mint a trace id for a turn.

    Uses the active span's trace id when tracing is on so the turn's
    id matches the span being exported. Falls back to a uuid4 hex
    when tracing is off and the active span is non-recording (its
    trace id is all-zeros). Either way a 32-char hex string.
    """
    span_context = trace.get_current_span().get_span_context()
    if span_context.trace_id != 0:
        return format(span_context.trace_id, "032x")
    return uuid4().hex


@contextmanager
def turn_trace() -> Iterator[str]:
    """Open a trace context for one session-service turn.

    Sets a freshly minted trace id for the duration of the block and
    resets it on exit, so nested or sequential turns each get their
    own id and the variable never leaks past the call. Yields the id
    in case the caller wants it directly.
    """
    trace_id = _new_trace_id()
    token = _current_trace_id.set(trace_id)
    try:
        yield trace_id
    finally:
        _current_trace_id.reset(token)


def current_trace_id() -> str | None:
    """Return the active turn's trace id, or None outside any turn.

    Read by the transport wrapper (to stamp llm_call rows) and the
    error logger (to stamp error_log rows). Both reading the same
    value is what links the two tables.
    """
    return _current_trace_id.get()
