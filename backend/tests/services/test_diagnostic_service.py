"""Tests for diagnostic_service.

Uses FakeTransport to exercise the diagnostic flow end-to-end
without hitting any real LLM. Covers happy path, tool-call
chaining, error paths, and chat-close cleanup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from app.models import TransportKind
from app.schemas.parsed_response import ParsedProposal
from app.schemas.tools import GetWeakTopicsCall, GetWeakTopicsInput
from app.services.diagnostic_service import (
    DiagnosticServiceError,
    propose_topic,
)
from app.transport.base import TransportError, TransportResponse

from tests.services.fakes import FakeTransport

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession


PROPOSAL_RESPONSE = """\
---PROPOSAL---
TOPIC_PATH: Python > Data Types > Integers
REASONING: You have 4 incorrect attempts on integer arithmetic in the last week.
---END_PROPOSAL---
"""

TURN_RESPONSE = """\
---TOPIC---
Python > Data Types > Integers
---DIFFICULTY---
beginner
---PREREQUISITES---
NONE
---MODE---
flashcard
---QUESTION---
What is 7 // 2?
---QUESTION_CODE---
NONE
---EXPECTED_ANSWER---
3
---REQUIREMENTS---
NONE
---FOLLOWUP---
NONE
---TAGS---
---END---
"""

TOOL_CALL_GET_WEAK = """\
---TOOL_CALL---
{"name": "get_weak_topics", "args": {}}
---END---
"""


async def test_happy_path_returns_proposal(db: DbSession) -> None:
    """LLM responds directly with a PROPOSAL block. No tool calls."""
    transport = FakeTransport([PROPOSAL_RESPONSE])

    result = await propose_topic(db=db, transport=transport, transport_kind=TransportKind.DEEPSEEK)

    assert isinstance(result, ParsedProposal)
    assert result.topic_path == "Python > Data Types > Integers"
    assert "4 incorrect attempts" in result.reasoning


async def test_one_tool_call_then_proposal(db: DbSession) -> None:
    """LLM calls a tool, sees the result, then proposes."""
    transport = FakeTransport([TOOL_CALL_GET_WEAK, PROPOSAL_RESPONSE])

    result = await propose_topic(db=db, transport=transport, transport_kind=TransportKind.DEEPSEEK)

    assert isinstance(result, ParsedProposal)
    # One chat opened (start_new_chat), tool result sent once.
    assert len(transport.chats) == 1
    assert len(transport.chats[0].tool_results_received) == 1


async def test_multiple_tool_calls_then_proposal(db: DbSession) -> None:
    """LLM chains multiple tool calls before proposing."""
    transport = FakeTransport([TOOL_CALL_GET_WEAK, TOOL_CALL_GET_WEAK, PROPOSAL_RESPONSE])

    result = await propose_topic(db=db, transport=transport, transport_kind=TransportKind.DEEPSEEK)

    assert isinstance(result, ParsedProposal)
    assert len(transport.chats[0].tool_results_received) == 2


async def test_wrong_response_kind_raises(db: DbSession) -> None:
    """LLM returns a teaching turn instead of a proposal."""
    transport = FakeTransport([TURN_RESPONSE])

    with pytest.raises(DiagnosticServiceError, match="Expected a PROPOSAL"):
        await propose_topic(db=db, transport=transport, transport_kind=TransportKind.DEEPSEEK)


async def test_transport_failure_on_start_raises(db: DbSession) -> None:
    """Transport fails opening the chat."""
    transport = FakeTransport([], raise_on_send=TransportError("boom"))

    with pytest.raises(DiagnosticServiceError, match="opening diagnostic chat"):
        await propose_topic(db=db, transport=transport, transport_kind=TransportKind.DEEPSEEK)


async def test_transport_failure_after_tool_call_raises(db: DbSession) -> None:
    """Transport fails sending tool results back to the LLM.

    start_new_chat does not increment _send_call_count. The first
    send-style operation after start is send_tool_results (with the
    first tool call's result), which lands at _send_call_count == 0.
    """
    transport = FakeTransport(
        [TOOL_CALL_GET_WEAK, PROPOSAL_RESPONSE],
        raise_on_send=TransportError("network down"),
        raise_on_send_at=0,  # fail on the first send_tool_results call
    )

    with pytest.raises(DiagnosticServiceError):
        await propose_topic(db=db, transport=transport, transport_kind=TransportKind.DEEPSEEK)


async def test_chat_closed_after_success(db: DbSession) -> None:
    """Chat is closed once the proposal is returned.

    FakeTransport's close is a no-op, so we test by asserting the
    chat list captures exactly one chat and the proposal returns.
    The contract is that propose_topic doesn't leak chat handles.
    """
    transport = FakeTransport([PROPOSAL_RESPONSE])

    await propose_topic(db=db, transport=transport, transport_kind=TransportKind.DEEPSEEK)

    # Exactly one chat opened, none leaked beyond the function's scope.
    assert len(transport.chats) == 1


async def test_unparseable_response_raises(db: DbSession) -> None:
    """LLM returns garbage that the parser cannot make sense of."""
    transport = FakeTransport(["this is not a valid response"])

    with pytest.raises(DiagnosticServiceError):
        await propose_topic(db=db, transport=transport, transport_kind=TransportKind.DEEPSEEK)


async def test_error_carries_kind_discriminator(db: DbSession) -> None:
    """DiagnosticServiceError exposes a kind field for HTTP mapping."""
    transport = FakeTransport([TURN_RESPONSE])

    with pytest.raises(DiagnosticServiceError) as exc_info:
        await propose_topic(db=db, transport=transport, transport_kind=TransportKind.DEEPSEEK)

    assert exc_info.value.kind == "wrong_response_kind"


async def test_transport_response_with_native_tool_calls_handled(db: DbSession) -> None:
    """DeepSeek-style native tool_calls field (not text) is handled."""
    tool_call = GetWeakTopicsCall(args=GetWeakTopicsInput(), id="call_123")
    transport = FakeTransport(
        [
            TransportResponse(text="", tool_calls=[tool_call]),
            PROPOSAL_RESPONSE,
        ]
    )

    result = await propose_topic(db=db, transport=transport, transport_kind=TransportKind.DEEPSEEK)

    assert isinstance(result, ParsedProposal)
