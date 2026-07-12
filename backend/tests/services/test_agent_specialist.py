"""Tests for agent_specialist: the retrieval specialist's loop.

FakeTransport drives the specialist loop without a real LLM. The
loop mirrors the planner's (open chat, tool-call loop, terminal
parse, close) with the specialist-specific behaviors under test: the
allowlist admits only search_corpus, retained search results are the
finding's evidence, and a finding without evidence is rejected.

execute_tool_call is monkeypatched at this module's namespace: the
real search_corpus handler runs a pgvector cosine query the SQLite
test engine cannot execute. The loop mechanics are what these tests
own; the real-search contract rides the milestone smoke.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from app.schemas.agent_specialist import SpecialistResult
from app.schemas.tools import (
    GetWeakTopicsCall,
    GetWeakTopicsInput,
    SearchCorpusCall,
    SearchCorpusInput,
    SearchCorpusOutput,
    SearchHitInfo,
    WeakTopicInfo,
)
from app.services import agent_specialist
from app.services.agent_specialist import SpecialistServiceError, gather_grounding
from app.services.tools.handlers import ToolHandlerError
from app.transport.base import TransportError, TransportResponse

from tests.services.fakes import FakeEmbedder, FakeTransport

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession

TARGET_PATH = "Python > Data Types > Integers"

FINDING = """\
---FINDING---
The corpus holds two learned items on integer division semantics.
---END---
"""

SEARCH_CALL = """\
---TOOL_CALL---
{"name": "search_corpus", "args": {"query": "integer division"}}
---END---
"""


def _weak_topic(path: str = TARGET_PATH) -> WeakTopicInfo:
    """The hand-off evidence row for the target."""
    return WeakTopicInfo(
        topic_path=path,
        incorrect_count=2,
        partial_count=0,
        correct_count=0,
        samples=[],
    )


def _search_output() -> SearchCorpusOutput:
    """A canned search result with one hit."""
    return SearchCorpusOutput(
        hits=[
            SearchHitInfo(
                source_type="learned_item",
                content="What is 7 // 2?",
                score=0.8,
            )
        ]
    )


@pytest.fixture
def fake_search(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace registry dispatch with a canned search result.

    Returns the list of dispatched tool names so tests can assert
    what reached the registry. Patched where agent_specialist
    resolves the name, per the patch-where-it-resolves rule.
    """
    dispatched: list[str] = []

    async def _fake_execute(db: DbSession, call: object, embedder: object) -> SearchCorpusOutput:
        dispatched.append(getattr(call, "name", "?"))
        return _search_output()

    monkeypatch.setattr(agent_specialist, "execute_tool_call", _fake_execute)
    return dispatched


# ---------- happy path ----------


async def test_search_then_finding_returns_grounded_result(
    db: DbSession, fake_search: list[str]
) -> None:
    """LLM searches, then emits a finding: result carries both halves.

    The full loop: chat opens with the curated hand-off, the search
    runs and its result is kept as evidence, the terminal finding
    parses, and the result pairs the finding with its evidence.
    """
    transport = FakeTransport([SEARCH_CALL, FINDING])

    result = await gather_grounding(
        db=db,
        transport=transport,
        embedder=FakeEmbedder(),
        topic_path=TARGET_PATH,
        weak_topic=_weak_topic(),
    )

    assert isinstance(result, SpecialistResult)
    assert result.finding.topic_path == TARGET_PATH
    assert result.finding.summary.startswith("The corpus holds")
    assert fake_search == ["search_corpus"]
    # The search result was retained as evidence.
    assert len(result.evidence) == 1
    assert result.evidence[0].tool == "search_corpus"
    # One chat opened, one tool result sent back.
    assert len(transport.chats) == 1
    assert len(transport.chats[0].tool_results_received) == 1
    # The chat advertises exactly the allowlisted surface, so the
    # transport can never offer a tool the gate would reject.
    assert transport.chats[0].tool_names == ("search_corpus",)


async def test_hand_off_message_carries_target_and_weak_data(
    db: DbSession, fake_search: list[str]
) -> None:
    """The first message holds the curated hand-off, nothing more.

    The specialist's world is the target path plus its weak-topic
    row. Both must reach the chat; the planner's conversation must
    not (the fake's messages are exactly intro + first message).
    """
    transport = FakeTransport([SEARCH_CALL, FINDING])

    await gather_grounding(
        db=db,
        transport=transport,
        embedder=FakeEmbedder(),
        topic_path=TARGET_PATH,
        weak_topic=_weak_topic(),
    )

    intro, first_message = transport.chats[0].messages_sent
    assert "FINDING" in intro
    assert TARGET_PATH in first_message
    assert '"incorrect_count":2' in first_message


async def test_native_tool_calls_then_finding(db: DbSession, fake_search: list[str]) -> None:
    """DeepSeek-style native tool_calls populate evidence the same way."""
    call = SearchCorpusCall(args=SearchCorpusInput(query="integer division"), id="call_1")
    transport = FakeTransport([TransportResponse(text="", tool_calls=[call]), FINDING])

    result = await gather_grounding(
        db=db,
        transport=transport,
        embedder=FakeEmbedder(),
        topic_path=TARGET_PATH,
        weak_topic=_weak_topic(),
    )

    assert len(result.evidence) == 1
    assert result.finding.topic_path == TARGET_PATH


# ---------- ungrounded finding ----------


async def test_finding_without_search_raises_ungrounded(
    db: DbSession, fake_search: list[str]
) -> None:
    """A finding emitted with no prior search has no evidence, so it dies.

    The LLM skips the tool call and jumps straight to a finding. The
    evidence list is empty, so the result is rejected. This makes
    "search first" structural rather than a prompt promise.
    """
    transport = FakeTransport([FINDING])

    with pytest.raises(SpecialistServiceError) as exc_info:
        await gather_grounding(
            db=db,
            transport=transport,
            embedder=FakeEmbedder(),
            topic_path=TARGET_PATH,
            weak_topic=_weak_topic(),
        )

    assert exc_info.value.kind == "ungrounded"
    assert fake_search == []


# ---------- disallowed tool ----------


async def test_disallowed_tool_call_raises_before_dispatch(
    db: DbSession, fake_search: list[str]
) -> None:
    """A tool outside the allowlist is rejected before the registry runs.

    get_weak_topics is a benign read on the planner's surface, but it
    is not on the specialist's, so the gate must kill it: the
    per-flow surface is the contract, not the tool's own safety.
    """
    bad_call = GetWeakTopicsCall(args=GetWeakTopicsInput(), id="call_x")
    transport = FakeTransport([TransportResponse(text="", tool_calls=[bad_call]), FINDING])

    with pytest.raises(SpecialistServiceError) as exc_info:
        await gather_grounding(
            db=db,
            transport=transport,
            embedder=FakeEmbedder(),
            topic_path=TARGET_PATH,
            weak_topic=_weak_topic(),
        )

    assert exc_info.value.kind == "disallowed_tool"
    # The disallowed tool never reached the registry.
    assert fake_search == []


# ---------- transport, parse, and handler failures ----------


async def test_transport_failure_on_start_raises(db: DbSession, fake_search: list[str]) -> None:
    """Transport fails opening the chat."""
    transport = FakeTransport([], raise_on_send=TransportError("boom"))

    with pytest.raises(SpecialistServiceError) as exc_info:
        await gather_grounding(
            db=db,
            transport=transport,
            embedder=FakeEmbedder(),
            topic_path=TARGET_PATH,
            weak_topic=_weak_topic(),
        )

    assert exc_info.value.kind == "transport_failed"


async def test_unparseable_terminal_raises_parse_failed(
    db: DbSession, fake_search: list[str]
) -> None:
    """LLM returns a teaching turn instead of a finding: parse failure.

    parse_specialist_response only accepts TOOL_CALL or FINDING, so a
    TOPIC turn dies inside it, same shape as the planner's grammar.
    """
    turn = "---TOPIC---\nx\n---DIFFICULTY---\nbeginner\n---END---\n"
    transport = FakeTransport([turn])

    with pytest.raises(SpecialistServiceError) as exc_info:
        await gather_grounding(
            db=db,
            transport=transport,
            embedder=FakeEmbedder(),
            topic_path=TARGET_PATH,
            weak_topic=_weak_topic(),
        )

    assert exc_info.value.kind == "parse_failed"


async def test_tool_handler_failure_raises(db: DbSession, monkeypatch: pytest.MonkeyPatch) -> None:
    """A handler failure surfaces as tool_handler_failed."""

    async def _raise_execute(db: DbSession, call: object, embedder: object) -> SearchCorpusOutput:
        raise ToolHandlerError("search backend down")

    monkeypatch.setattr(agent_specialist, "execute_tool_call", _raise_execute)
    transport = FakeTransport([SEARCH_CALL, FINDING])

    with pytest.raises(SpecialistServiceError) as exc_info:
        await gather_grounding(
            db=db,
            transport=transport,
            embedder=FakeEmbedder(),
            topic_path=TARGET_PATH,
            weak_topic=_weak_topic(),
        )

    assert exc_info.value.kind == "tool_handler_failed"
