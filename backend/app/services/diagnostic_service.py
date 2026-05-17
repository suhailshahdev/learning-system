"""Diagnostic service.

Drives the "what should I focus on today?" workflow.
The LLM reads analytical state via tools and proposes
one topic. The user accepts or rejects. The chat is
throwaway with no persistence or session row, just
a one-shot LLM call.

This service is the only consumer of the diagnostic intro.
The LLM-facing surface ends here: the route layer calls
propose_topic and returns the result as JSON.
"""

from __future__ import annotations

import contextlib
import json
from typing import TYPE_CHECKING, Any, Literal

from sqlalchemy import select

from app.models import Domain, Topic
from app.prompts.diagnostic_intro import build_diagnostic_intro
from app.schemas.parsed_response import ParsedProposal, ParsedToolCall
from app.services.parser import parse_response
from app.services.tools.handlers import ToolHandlerError
from app.services.tools.registry import execute_tool_call
from app.transport.base import (
    ToolResult,
    TransportError,
    TransportResponse,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession

    from app.models import TransportKind
    from app.schemas.parsed_response import ParsedResponse
    from app.transport.base import LLMTransport


# First message sent to the diagnostic chat after the intro.
# Kept thin: the intro carries the instructions, this is the
# "go" signal.
_FIRST_MESSAGE = (
    "Read the user's analytical state using the tools available "
    "to you, then propose one topic for them to focus on. Respond "
    "with a PROPOSAL block as specified in the intro."
)


# Failure modes for diagnostic_service. The route layer maps these
# to HTTP status codes. Substring matching on messages was
# considered and rejected: messages drift, kinds don't.
type DiagnosticErrorKind = Literal[
    "transport_failed",
    "parse_failed",
    "wrong_response_kind",
    "tool_handler_failed",
    "no_data",
    "unexpected",
]


class DiagnosticServiceError(Exception):
    """A diagnostic-service operation failed.

    Wraps the underlying cause so callers see one error type at the
    service boundary. Distinct from SessionServiceError because
    diagnostic mode is not a session-service operation: there is
    no session row, no persistence, no resume.

    kind is the discriminator the route layer uses to pick the
    right HTTP status code. The pattern mirrors SessionResumeError.
    """

    def __init__(
        self,
        message: str,
        kind: DiagnosticErrorKind,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.kind: DiagnosticErrorKind = kind
        self.cause = cause


async def propose_topic(
    *,
    db: DbSession,
    transport: LLMTransport[Any],
    transport_kind: TransportKind,
) -> ParsedProposal:
    """Open a diagnostic chat, drive tool calls to completion, return the proposal.

    Builds the diagnostic intro from current DB state, opens a
    fresh chat on the transport, drives the tool-call loop until
    a terminal response lands, validates it's a ParsedProposal,
    closes the chat, and returns the proposal.

    Guards against the empty-state case before any transport call:
    if no domains exist or no topics exist, raises with
    kind="no_data". The LLM has been observed proposing the
    "(none yet)" placeholder string as a topic_path when the intro
    has nothing real to offer. Prompt-level rules don't hold
    against this. A service-layer guard is the load-bearing fix.

    The chat is throwaway. No persistence happens. Tool
    handlers commit their own writes (the diagnostic tools
    are read-only so this is moot, but the contract holds).

    Raises DiagnosticServiceError on any failure: empty state,
    transport error, parse error, wrong response kind, tool
    handler failure, or close failure.
    """
    _check_diagnosable_state(db)

    intro = await build_diagnostic_intro(db)

    try:
        chat, response = await transport.start_new_chat(intro, _FIRST_MESSAGE)
    except TransportError as e:
        raise DiagnosticServiceError(
            f"Transport failed opening diagnostic chat: {e.message}",
            kind="transport_failed",
            cause=e,
        ) from e

    try:
        parsed = _response_to_parsed(response)
        response, parsed = await _execute_until_proposal(
            transport=transport,
            chat=chat,
            response=response,
            parsed=parsed,
            db=db,
        )
    except DiagnosticServiceError:
        # Close the chat even on failure. Diagnostic chats are throwaway
        # and leaving one open leaks transport state.
        await _close_quietly(transport, chat)
        raise
    except Exception as e:
        await _close_quietly(transport, chat)
        raise DiagnosticServiceError(
            f"Unexpected error during diagnostic flow: {e}",
            kind="unexpected",
            cause=e,
        ) from e

    await _close_quietly(transport, chat)

    if not isinstance(parsed, ParsedProposal):
        raise DiagnosticServiceError(
            f"Expected a PROPOSAL response, got {parsed.kind!r}.",
            kind="wrong_response_kind",
        )
    return parsed


async def _execute_until_proposal(
    *,
    transport: LLMTransport[Any],
    chat: Any,
    response: TransportResponse,
    parsed: ParsedResponse,
    db: DbSession,
) -> tuple[TransportResponse, ParsedResponse]:
    """Drive tool calls until the LLM produces a non-tool-call response.

    Mirrors the session service tool loop but without persistence
    since no session row exists. Each tool call goes through the same
    registry as the teaching flow and results are sent back via
    send_tool_results.

    transport_kind is accepted for symmetry with the session helper
    but unused since there is no per-transport branching in the
    diagnostic flow.

    Returns the final (response, parsed) when parsed is no longer
    a ParsedToolCall.
    """
    while isinstance(parsed, ParsedToolCall):
        # Execute every tool call in the response before sending
        # results back. OpenAI-compatible APIs require all calls
        # in one assistant message to be answered together.
        # Diagnostic mode is read-only, so handler failures here
        # mean a real bug rather than a write-conflict.
        results: list[ToolResult] = []
        for call in parsed.calls:
            try:
                output = await execute_tool_call(db, call)
            except ToolHandlerError as e:
                raise DiagnosticServiceError(
                    f"Tool handler {call.name!r} failed: {e.message}",
                    kind="tool_handler_failed",
                    cause=e,
                ) from e
            content = output.model_dump_json()
            results.append(ToolResult(call_id=call.id or call.name, content=content))

        try:
            response = await transport.send_tool_results(chat, results)
        except TransportError as e:
            raise DiagnosticServiceError(
                f"Transport failed sending tool results: {e.message}",
                kind="transport_failed",
                cause=e,
            ) from e

        try:
            parsed = _response_to_parsed(response)
        except Exception as e:
            raise DiagnosticServiceError(
                "Parse failed on response after tool result.",
                kind="parse_failed",
                cause=e,
            ) from e

    return response, parsed


def _response_to_parsed(response: TransportResponse) -> ParsedResponse:
    """Translate a TransportResponse into a ParsedResponse.

    Same shape as session_service._response_to_parsed: tool_calls
    field takes precedence (DeepSeek native function calling),
    otherwise parse the text (Claude transport's TOOL_CALL block
    or a terminal PROPOSAL).

    All tool_calls in the response are passed through. OpenAI-
    compatible APIs require every tool_call_id to be answered
    in the next request.
    """
    if response.tool_calls:
        calls = list(response.tool_calls)
        raw_text = json.dumps([c.model_dump(mode="json") for c in calls])
        return ParsedToolCall(calls=calls, raw_text=raw_text)
    return parse_response(response.text)


async def _close_quietly(transport: LLMTransport[Any], chat: Any) -> None:
    """Close the chat, swallowing any error from the close itself.

    A failed close is not worth promoting over the original error
    being raised. Throwaway chats: leaking the chat handle is
    acceptable if close fails. The next chat's start will surface
    any persistent transport problem.
    """
    with contextlib.suppress(Exception):
        await transport.close(chat)


def _check_diagnosable_state(db: DbSession) -> None:
    """Raise DiagnosticServiceError(kind="no_data") if the DB has nothing to diagnose.

    Empty state is defined as no domains OR no topics. Both
    conditions produce an unactionable proposal: the LLM has no
    real path to suggest, but historically has emitted the
    placeholder string from the intro as a path anyway.

    Topics-but-no-domains is also empty: the diagnostic intro pulls
    list_domains to build its EXISTING DOMAINS section. An empty
    result there reproduces the placeholder-as-topic bug. Although
    Topic.domain is denormalized so the data model permits orphan
    topics, the diagnostic intro requires real Domain rows.

    Domains-but-no-topics is empty because the intro's own rule
    says proposals must be paths that exist in the topic tree.
    """
    has_domain = db.execute(select(Domain.id).limit(1)).scalar_one_or_none() is not None
    has_topic = db.execute(select(Topic.id).limit(1)).scalar_one_or_none() is not None

    if not has_topic:
        raise DiagnosticServiceError(
            "No topics exist yet. Start a learning session first to build diagnosable history.",
            kind="no_data",
        )
    if not has_domain:
        raise DiagnosticServiceError(
            "No domains exist yet. Seed the domain table or start a session to register a domain.",
            kind="no_data",
        )
