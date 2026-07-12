"""Retrieval specialist service.

Runs one specialist invocation: a throwaway LLM chat that searches
the corpus for material related to a single plan target and returns
a grounding finding plus the search evidence behind it. The caller
(the planner's propose flow) invokes it once per plan target after
the plan has parsed and passed groundedness; the specialist never
sees the planner conversation and never mutates anything.

Mirrors the planner's loop shape: tool calls are gated against a
read allowlist before registry dispatch, the same tuple is the
chat's advertised surface, and tool results are retained as
Evidence. The terminal is a FINDING block instead of a PLAN.
"""

from __future__ import annotations

import contextlib
import json
from typing import TYPE_CHECKING, Any, Literal

from app.prompts.retrieval_specialist_intro import build_retrieval_specialist_intro
from app.schemas.agent_plan import Evidence
from app.schemas.agent_specialist import SpecialistFinding, SpecialistResult
from app.schemas.parsed_response import ParsedToolCall
from app.services.parser import ParseError, parse_specialist_response
from app.services.tools.handlers import ToolHandlerError
from app.services.tools.registry import execute_tool_call
from app.transport.base import (
    ToolResult,
    TransportError,
    TransportResponse,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession

    from app.schemas.agent_specialist import ParsedFinding
    from app.schemas.tools import WeakTopicInfo
    from app.services.embedding_service import Embedder
    from app.transport.base import LLMTransport


# Tools the specialist LLM may call during its loop. Checked before
# registry dispatch and advertised as the chat's native surface, so
# the transport can never offer a tool this gate would reject.
_SPECIALIST_TOOL_NAMES: tuple[str, ...] = ("search_corpus",)
_ALLOWED_TOOLS = frozenset(_SPECIALIST_TOOL_NAMES)


# Failure modes for the specialist service. no_data is absent: the
# caller only invokes the specialist for targets that already passed
# the planner's guards, so there is no empty state to detect here.
# ungrounded means the LLM emitted a finding without searching first.
type SpecialistErrorKind = Literal[
    "transport_failed",
    "parse_failed",
    "tool_handler_failed",
    "disallowed_tool",
    "ungrounded",
    "unexpected",
]


class SpecialistServiceError(Exception):
    """A specialist invocation failed.

    Wraps the underlying cause so callers see one error type at the
    service boundary. kind is the discriminator: the same cluster
    meanings as the planner's kinds, so a caller mapping failures to
    HTTP statuses can reuse the planner's dispatch shape.
    """

    def __init__(
        self,
        message: str,
        kind: SpecialistErrorKind,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.kind: SpecialistErrorKind = kind
        self.cause = cause


async def gather_grounding(
    *,
    db: DbSession,
    transport: LLMTransport[Any],
    embedder: Embedder,
    topic_path: str,
    weak_topic: WeakTopicInfo,
) -> SpecialistResult:
    """Run one retrieval-specialist chat for a single plan target.

    Opens a fresh chat with the static intro and the curated hand-off
    context (the target path and its weak-topic evidence row, nothing
    else), drives the tool-call loop keeping search_corpus results as
    Evidence, parses the terminal finding, and requires at least one
    retained result so a finding cannot be emitted from nothing.

    Read-only: the specialist's allowlist holds no mutating tool and
    the gate dies structurally on any attempt to reach one.

    Raises SpecialistServiceError on transport failure, parse
    failure, a disallowed tool call, a tool handler failure, or a
    finding emitted without evidence.
    """
    intro = build_retrieval_specialist_intro()
    first_message = _build_first_message(topic_path, weak_topic)

    try:
        chat, response = await transport.start_new_chat(
            intro, first_message, tool_names=_SPECIALIST_TOOL_NAMES
        )
    except TransportError as e:
        raise SpecialistServiceError(
            f"Transport failed opening specialist chat: {e.message}",
            kind="transport_failed",
            cause=e,
        ) from e

    try:
        parsed, evidence = await _execute_until_finding(
            transport=transport,
            embedder=embedder,
            chat=chat,
            response=response,
            db=db,
        )
    except SpecialistServiceError:
        # Close the chat even on failure. Specialist chats are
        # throwaway and leaving one open leaks transport state.
        await _close_quietly(transport, chat)
        raise
    except Exception as e:
        await _close_quietly(transport, chat)
        raise SpecialistServiceError(
            f"Unexpected error during specialist flow: {e}",
            kind="unexpected",
            cause=e,
        ) from e

    await _close_quietly(transport, chat)

    if not evidence:
        raise SpecialistServiceError(
            "Specialist emitted a finding without any search evidence.",
            kind="ungrounded",
        )

    finding = SpecialistFinding(topic_path=topic_path, summary=parsed.summary)
    return SpecialistResult(finding=finding, evidence=evidence)


def _build_first_message(topic_path: str, weak_topic: WeakTopicInfo) -> str:
    """Build the hand-off message: the target and its weak-topic row.

    This is the specialist's entire view of the world beyond its own
    tool calls. Deliberately minimal: no planner conversation, no
    other targets, no plan. The weak-topic row rides along so the
    specialist can shape its queries around the observed misses.
    """
    return (
        f"Target topic: {topic_path}\n"
        f"Weak-topic data: {weak_topic.model_dump_json()}\n\n"
        "Search the corpus for material related to this topic, then "
        "respond with a FINDING block."
    )


async def _execute_until_finding(
    *,
    transport: LLMTransport[Any],
    embedder: Embedder,
    chat: Any,
    response: TransportResponse,
    db: DbSession,
) -> tuple[ParsedFinding, list[Evidence]]:
    """Drive tool calls until the LLM emits the terminal finding.

    Every call is checked against the allowlist before dispatch, then
    runs through the same registry as the other flows. Results from
    search_corpus are kept as Evidence. The caller requires at least
    one before accepting the finding.
    """
    evidence: list[Evidence] = []
    parsed = _parse_or_raise(response)
    while isinstance(parsed, ParsedToolCall):
        # Execute every call in the response before sending results
        # back. OpenAI-compatible APIs require all calls in one
        # assistant message to be answered together.
        results: list[ToolResult] = []
        for call in parsed.calls:
            if call.name not in _ALLOWED_TOOLS:
                raise SpecialistServiceError(
                    f"Tool {call.name!r} is not available to the specialist.",
                    kind="disallowed_tool",
                )
            try:
                output = await execute_tool_call(db, call, embedder)
            except ToolHandlerError as e:
                raise SpecialistServiceError(
                    f"Tool handler {call.name!r} failed: {e.message}",
                    kind="tool_handler_failed",
                    cause=e,
                ) from e
            evidence.append(Evidence(tool=call.name, result=output.model_dump(mode="json")))
            results.append(
                ToolResult(call_id=call.id or call.name, content=output.model_dump_json())
            )

        try:
            response = await transport.send_tool_results(chat, results)
        except TransportError as e:
            raise SpecialistServiceError(
                f"Transport failed sending tool results: {e.message}",
                kind="transport_failed",
                cause=e,
            ) from e

        parsed = _parse_or_raise(response)

    return parsed, evidence


def _parse_or_raise(response: TransportResponse) -> ParsedToolCall | ParsedFinding:
    """Translate a TransportResponse, wrapping parse failures uniformly."""
    try:
        return _response_to_parsed(response)
    except ParseError as e:
        raise SpecialistServiceError(
            f"Parse failed on specialist response: {e.message}",
            kind="parse_failed",
            cause=e,
        ) from e


def _response_to_parsed(response: TransportResponse) -> ParsedToolCall | ParsedFinding:
    """Translate a TransportResponse for the specialist flow.

    Same shape as the planner's translation: native tool_calls take
    precedence (DeepSeek), otherwise the text parses through the
    specialist grammar, a TOOL_CALL block or the terminal FINDING.
    """
    if response.tool_calls:
        calls = list(response.tool_calls)
        raw_text = json.dumps([c.model_dump(mode="json") for c in calls])
        return ParsedToolCall(calls=calls, raw_text=raw_text)
    return parse_specialist_response(response.text)


async def _close_quietly(transport: LLMTransport[Any], chat: Any) -> None:
    """Close the chat, swallowing any error from the close itself.

    A failed close is not worth promoting over the original error
    being raised. Throwaway chats: leaking the chat handle is
    acceptable if close fails.
    """
    with contextlib.suppress(Exception):
        await transport.close(chat)
