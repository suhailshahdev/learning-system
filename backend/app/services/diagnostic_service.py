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
from typing import TYPE_CHECKING, Any

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


class DiagnosticServiceError(Exception):
    """A diagnostic-service operation failed.

    Wraps the underlying cause so callers see one error type at the
    service boundary. Distinct from SessionServiceError because
    diagnostic mode is not a session-service operation: there is
    no session row, no persistence, no resume.
    """

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.message = message
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

    The chat is throwaway. No persistence happens. Tool
    handlers commit their own writes (the diagnostic tools
    are read-only so this is moot, but the contract holds).

    Raises DiagnosticServiceError on any failure: transport error,
    parse error, wrong response kind, tool handler failure, or
    close failure.
    """
    intro = await build_diagnostic_intro(db)

    try:
        chat, response = await transport.start_new_chat(intro, _FIRST_MESSAGE)
    except TransportError as e:
        raise DiagnosticServiceError(
            f"Transport failed opening diagnostic chat: {e.message}", cause=e
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
            f"Unexpected error during diagnostic flow: {e}", cause=e
        ) from e

    await _close_quietly(transport, chat)

    if not isinstance(parsed, ParsedProposal):
        raise DiagnosticServiceError(f"Expected a PROPOSAL response, got {parsed.kind!r}.")
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
        try:
            output = await execute_tool_call(db, parsed.call)
        except ToolHandlerError as e:
            raise DiagnosticServiceError(
                f"Tool handler {parsed.call.name!r} failed: {e.message}", cause=e
            ) from e

        content = output.model_dump_json()
        call_id = parsed.call.id or parsed.call.name
        result = ToolResult(call_id=call_id, content=content)

        try:
            response = await transport.send_tool_results(chat, [result])
        except TransportError as e:
            raise DiagnosticServiceError(
                f"Transport failed sending tool results: {e.message}", cause=e
            ) from e

        try:
            parsed = _response_to_parsed(response)
        except Exception as e:
            raise DiagnosticServiceError(
                "Parse failed on response after tool result.", cause=e
            ) from e

    return response, parsed


def _response_to_parsed(response: TransportResponse) -> ParsedResponse:
    """Translate a TransportResponse into a ParsedResponse.

    Same shape as session_service._response_to_parsed: tool_calls
    field takes precedence (DeepSeek native function calling),
    otherwise parse the text (Claude transport's TOOL_CALL block
    or a terminal PROPOSAL).
    """
    if response.tool_calls:
        call = response.tool_calls[0]
        return ParsedToolCall(call=call, raw_text=call.model_dump_json())
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
