"""Tests for parse_specialist_response, the specialist-flow grammar.

Specialist conversations parse through the same two-phase grammar
shape as the planner: a TOOL_CALL block mid-conversation or a
terminal FINDING block. FINDING bodies are plain prose, not JSON.
These tests pin the prose wire form and the loud-failure cases.
"""

from __future__ import annotations

import pytest
from app.schemas.agent_specialist import ParsedFinding
from app.schemas.parsed_response import ParsedToolCall
from app.services.parser import ParseError, parse_specialist_response

FINDING = """\
---FINDING---
The corpus holds two learned items on list appends and one note on
extend versus append semantics.
---END---
"""

TOOL_CALL = """\
---TOOL_CALL---
{"name": "search_corpus", "args": {"query": "list append"}}
---END---
"""


def test_terminal_finding_parses_to_parsed_finding() -> None:
    """A FINDING block yields a ParsedFinding carrying the prose body."""
    result = parse_specialist_response(FINDING)

    assert isinstance(result, ParsedFinding)
    assert result.summary.startswith("The corpus holds two learned items")
    assert result.raw_text == result.summary


def test_multiline_finding_body_is_preserved() -> None:
    """Line breaks inside the prose body survive parsing."""
    result = parse_specialist_response(FINDING)

    assert isinstance(result, ParsedFinding)
    assert "\n" in result.summary


def test_tool_call_parses_to_parsed_tool_call() -> None:
    """A TOOL_CALL block routes to the existing tool-call parser."""
    result = parse_specialist_response(TOOL_CALL)

    assert isinstance(result, ParsedToolCall)
    assert result.calls[0].name == "search_corpus"


def test_teaching_leading_marker_rejected() -> None:
    """A TOPIC turn is not valid in the specialist grammar.

    parse_response accepts TOPIC, parse_specialist_response must not,
    so a wrong-kind terminal from the specialist LLM dies as a parse
    failure rather than being routed anywhere.
    """
    with pytest.raises(ParseError, match="specialist response"):
        parse_specialist_response("---TOPIC---\nx\n---END---\n")


def test_plan_leading_marker_rejected() -> None:
    """The planner's terminal is not valid in the specialist grammar."""
    with pytest.raises(ParseError, match="specialist response"):
        parse_specialist_response('---PLAN---\n[{"tool": "x"}]\n---END---\n')


def test_finding_without_end_raises() -> None:
    """A FINDING block missing its END marker raises."""
    with pytest.raises(ParseError, match="followed by END"):
        parse_specialist_response("---FINDING---\nSome note.\n")


def test_empty_finding_body_raises() -> None:
    """An empty FINDING body raises: nothing-found is said in words."""
    with pytest.raises(ParseError, match="empty"):
        parse_specialist_response("---FINDING---\n---END---\n")


def test_no_delimiters_raises() -> None:
    """Bare prose without delimiters raises."""
    with pytest.raises(ParseError, match="No delimiters"):
        parse_specialist_response("The corpus holds nothing.")
