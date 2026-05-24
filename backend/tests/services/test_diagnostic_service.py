"""Tests for diagnostic_service.

Uses FakeTransport to exercise the diagnostic flow end-to-end
without hitting any real LLM. Covers happy path, tool-call
chaining, error paths, and chat-close cleanup.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from app.models import (
    Domain,
    DomainKind,
    Topic,
    TopicStatus,
    TransportKind,
)
from app.schemas.parsed_response import ParsedProposal
from app.schemas.tools import GetWeakTopicsCall, GetWeakTopicsInput
from app.services.diagnostic_service import (
    DiagnosticServiceError,
    propose_topic,
)
from app.transport.base import TransportError, TransportResponse

from tests.services.fakes import FakeEmbedder, FakeTransport

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession


def _add_domain(db: DbSession, name: str = "Python") -> Domain:
    """Add a minimal Domain row. Used by the diagnosable-state fixture
    and by empty-state tests that need to seed selectively."""
    domain = Domain(name=name, kind=DomainKind.LANGUAGE, description=None)
    db.add(domain)
    db.flush()
    return domain


def _add_topic(db: DbSession, path: str = "Python > Basics") -> Topic:
    """Add a minimal Topic row. Used by the diagnosable-state fixture
    and by empty-state tests that need to seed selectively."""
    topic = Topic(
        path=path,
        domain=path.split(" > ", 1)[0],
        name=path.rsplit(" > ", 1)[-1],
        status=TopicStatus.IN_PROGRESS,
    )
    db.add(topic)
    db.flush()
    return topic


@pytest.fixture
def diagnosable_db(db: DbSession) -> DbSession:
    """Seed the minimal state needed for propose_topic to proceed.

    The empty-state guard requires both a Domain and a Topic row.
    Most tests exercise the post-guard flow and need this baseline.
    Tests that specifically exercise the guard itself accept the
    bare `db` fixture instead.
    """
    _add_domain(db)
    _add_topic(db)
    return db


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


async def test_happy_path_returns_proposal(diagnosable_db: DbSession) -> None:
    """LLM responds directly with a PROPOSAL block. No tool calls."""
    transport = FakeTransport([PROPOSAL_RESPONSE])

    result = await propose_topic(
        db=diagnosable_db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        embedder=FakeEmbedder(),
    )

    assert isinstance(result, ParsedProposal)
    assert result.topic_path == "Python > Data Types > Integers"
    assert "4 incorrect attempts" in result.reasoning


async def test_one_tool_call_then_proposal(diagnosable_db: DbSession) -> None:
    """LLM calls a tool, sees the result, then proposes."""
    transport = FakeTransport([TOOL_CALL_GET_WEAK, PROPOSAL_RESPONSE])

    result = await propose_topic(
        db=diagnosable_db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        embedder=FakeEmbedder(),
    )

    assert isinstance(result, ParsedProposal)
    # One chat opened (start_new_chat), tool result sent once.
    assert len(transport.chats) == 1
    assert len(transport.chats[0].tool_results_received) == 1


async def test_multiple_tool_calls_then_proposal(diagnosable_db: DbSession) -> None:
    """LLM chains multiple tool calls before proposing."""
    transport = FakeTransport([TOOL_CALL_GET_WEAK, TOOL_CALL_GET_WEAK, PROPOSAL_RESPONSE])

    result = await propose_topic(
        db=diagnosable_db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        embedder=FakeEmbedder(),
    )

    assert isinstance(result, ParsedProposal)
    assert len(transport.chats[0].tool_results_received) == 2


async def test_wrong_response_kind_raises(diagnosable_db: DbSession) -> None:
    """LLM returns a teaching turn instead of a proposal."""
    transport = FakeTransport([TURN_RESPONSE])

    with pytest.raises(DiagnosticServiceError, match="Expected a PROPOSAL"):
        await propose_topic(
            db=diagnosable_db,
            transport=transport,
            transport_kind=TransportKind.DEEPSEEK,
            embedder=FakeEmbedder(),
        )


async def test_transport_failure_on_start_raises(diagnosable_db: DbSession) -> None:
    """Transport fails opening the chat."""
    transport = FakeTransport([], raise_on_send=TransportError("boom"))

    with pytest.raises(DiagnosticServiceError, match="opening diagnostic chat"):
        await propose_topic(
            db=diagnosable_db,
            transport=transport,
            transport_kind=TransportKind.DEEPSEEK,
            embedder=FakeEmbedder(),
        )


async def test_transport_failure_after_tool_call_raises(diagnosable_db: DbSession) -> None:
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
        await propose_topic(
            db=diagnosable_db,
            transport=transport,
            transport_kind=TransportKind.DEEPSEEK,
            embedder=FakeEmbedder(),
        )


async def test_chat_closed_after_success(diagnosable_db: DbSession) -> None:
    """Chat is closed once the proposal is returned.

    FakeTransport's close is a no-op, so we test by asserting the
    chat list captures exactly one chat and the proposal returns.
    The contract is that propose_topic doesn't leak chat handles.
    """
    transport = FakeTransport([PROPOSAL_RESPONSE])

    await propose_topic(
        db=diagnosable_db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        embedder=FakeEmbedder(),
    )

    # Exactly one chat opened, none leaked beyond the function's scope.
    assert len(transport.chats) == 1


async def test_unparseable_response_raises(diagnosable_db: DbSession) -> None:
    """LLM returns garbage that the parser cannot make sense of."""
    transport = FakeTransport(["this is not a valid response"])

    with pytest.raises(DiagnosticServiceError):
        await propose_topic(
            db=diagnosable_db,
            transport=transport,
            transport_kind=TransportKind.DEEPSEEK,
            embedder=FakeEmbedder(),
        )


async def test_error_carries_kind_discriminator(diagnosable_db: DbSession) -> None:
    """DiagnosticServiceError exposes a kind field for HTTP mapping."""
    transport = FakeTransport([TURN_RESPONSE])

    with pytest.raises(DiagnosticServiceError) as exc_info:
        await propose_topic(
            db=diagnosable_db,
            transport=transport,
            transport_kind=TransportKind.DEEPSEEK,
            embedder=FakeEmbedder(),
        )

    assert exc_info.value.kind == "wrong_response_kind"


async def test_transport_response_with_native_tool_calls_handled(diagnosable_db: DbSession) -> None:
    """DeepSeek-style native tool_calls field (not text) is handled."""
    tool_call = GetWeakTopicsCall(args=GetWeakTopicsInput(), id="call_123")
    transport = FakeTransport(
        [
            TransportResponse(text="", tool_calls=[tool_call]),
            PROPOSAL_RESPONSE,
        ]
    )

    result = await propose_topic(
        db=diagnosable_db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        embedder=FakeEmbedder(),
    )

    assert isinstance(result, ParsedProposal)


async def test_multiple_native_tool_calls_in_one_response_all_executed(
    diagnosable_db: DbSession,
) -> None:
    """DeepSeek returning N tool_calls in one response: all N execute, all N results sent back.

    Falsifying test for the bug where the service only executed
    tool_calls[0] and the DeepSeek API rejected the next request
    because fewer-than-N tool messages followed an N-call assistant
    message.
    """
    weak_call = GetWeakTopicsCall(args=GetWeakTopicsInput(), id="call_001")
    stale_call = GetWeakTopicsCall(args=GetWeakTopicsInput(), id="call_002")

    transport = FakeTransport(
        [
            TransportResponse(text="", tool_calls=[weak_call, stale_call]),
            PROPOSAL_RESPONSE,
        ]
    )

    result = await propose_topic(
        db=diagnosable_db,
        transport=transport,
        transport_kind=TransportKind.DEEPSEEK,
        embedder=FakeEmbedder(),
    )

    assert isinstance(result, ParsedProposal)
    # Both calls' results must have been bundled into one send_tool_results
    # call. If only one was sent, this is the regression.
    assert len(transport.chats) == 1
    assert len(transport.chats[0].tool_results_received) == 1
    results_sent = transport.chats[0].tool_results_received[0]
    assert len(results_sent) == 2
    sent_ids = {r.call_id for r in results_sent}
    assert sent_ids == {"call_001", "call_002"}


# ---------- empty-state guard ----------


async def test_empty_db_raises_no_data_without_transport_call(db: DbSession) -> None:
    """Falsifying test: empty DB → guard fires → transport never called.

    The bug was that the LLM proposed the placeholder string from
    the intro as a topic. The fix is a pre-transport guard. This
    test asserts the guard prevents the LLM call entirely, not
    just that it filters the response after.
    """
    transport = FakeTransport([PROPOSAL_RESPONSE])

    with pytest.raises(DiagnosticServiceError) as exc_info:
        await propose_topic(
            db=db,
            transport=transport,
            transport_kind=TransportKind.DEEPSEEK,
            embedder=FakeEmbedder(),
        )

    assert exc_info.value.kind == "no_data"
    # Critical: transport was never opened. If chats is non-empty,
    # the guard fired too late.
    assert len(transport.chats) == 0


async def test_domain_only_no_topics_raises_no_data(db: DbSession) -> None:
    """Domain rows exist but no topics: still unactionable, still no_data."""
    _add_domain(db)

    transport = FakeTransport([PROPOSAL_RESPONSE])

    with pytest.raises(DiagnosticServiceError) as exc_info:
        await propose_topic(
            db=db,
            transport=transport,
            transport_kind=TransportKind.DEEPSEEK,
            embedder=FakeEmbedder(),
        )

    assert exc_info.value.kind == "no_data"
    assert "topics" in exc_info.value.message.lower()
    assert len(transport.chats) == 0


async def test_topic_without_domain_row_raises_no_data(db: DbSession) -> None:
    """Topic exists but Domain table empty: still no_data.

    Although Topic.domain is denormalized so the data model
    permits orphan-domain topics, the diagnostic intro pulls
    list_domains to build its EXISTING DOMAINS section. An empty
    list_domains result reproduces the original placeholder-as-
    topic bug. The guard requires both a Domain row and a Topic
    row before proceeding.
    """
    _add_topic(db)

    transport = FakeTransport([PROPOSAL_RESPONSE])

    with pytest.raises(DiagnosticServiceError) as exc_info:
        await propose_topic(
            db=db,
            transport=transport,
            transport_kind=TransportKind.DEEPSEEK,
            embedder=FakeEmbedder(),
        )

    assert exc_info.value.kind == "no_data"
    assert "domains" in exc_info.value.message.lower()
    # Transport should not have been opened.
    assert len(transport.chats) == 0


async def test_topic_and_domain_present_proceeds_normally(db: DbSession) -> None:
    """Both tables populated: full normal flow runs."""
    _add_domain(db)
    _add_topic(db)

    transport = FakeTransport([PROPOSAL_RESPONSE])

    result = await propose_topic(
        db=db, transport=transport, transport_kind=TransportKind.DEEPSEEK, embedder=FakeEmbedder()
    )

    assert isinstance(result, ParsedProposal)
    assert len(transport.chats) == 1


async def test_no_data_error_carries_kind_discriminator(db: DbSession) -> None:
    """Route layer dispatches on kind, not message. Confirm kind is set."""
    transport = FakeTransport([PROPOSAL_RESPONSE])

    with pytest.raises(DiagnosticServiceError) as exc_info:
        await propose_topic(
            db=db,
            transport=transport,
            transport_kind=TransportKind.DEEPSEEK,
            embedder=FakeEmbedder(),
        )

    assert exc_info.value.kind == "no_data"
    assert isinstance(exc_info.value.message, str)
    assert len(exc_info.value.message) > 0
