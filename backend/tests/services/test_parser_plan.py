"""Tests for parse_plan_response, the planner-flow grammar.

The planner conversation parses through a narrower grammar than the
teaching flow: a TOOL_CALL block mid-conversation or a terminal PLAN
block. PLAN bodies are JSON arrays of mutate steps. These tests pin
the array-only wire form, per-entry validation against the mutate
vocabulary, and the loud-failure cases the format depends on.
"""

from __future__ import annotations

import pytest
from app.schemas.agent_plan import MarkForRevisionStep, ParsedPlan
from app.schemas.parsed_response import ParsedToolCall
from app.services.parser import ParseError, parse_plan_response

ONE_STEP_PLAN = """\
---PLAN---
[{"tool": "mark_for_revision", "args": {"path": "Python > Async > Tasks"}}]
---END---
"""

TWO_STEP_PLAN = """\
---PLAN---
[
  {"tool": "mark_for_revision", "args": {"path": "Python > Async > Tasks"}},
  {"tool": "mark_for_revision", "args": {"path": "Python > Async > Gather"}}
]
---END---
"""

TOOL_CALL = """\
---TOOL_CALL---
{"name": "get_weak_topics", "args": {}}
---END---
"""


def test_terminal_plan_parses_to_parsed_plan() -> None:
    """A PLAN block yields a ParsedPlan with one mutate step."""
    result = parse_plan_response(ONE_STEP_PLAN)

    assert isinstance(result, ParsedPlan)
    assert len(result.plan.steps) == 1
    step = result.plan.steps[0]
    assert isinstance(step, MarkForRevisionStep)
    assert step.args.path == "Python > Async > Tasks"
    assert result.raw_text


def test_multi_step_plan_parses_in_order() -> None:
    """A multi-entry PLAN array preserves entry order."""
    result = parse_plan_response(TWO_STEP_PLAN)

    assert isinstance(result, ParsedPlan)
    paths = [s.args.path for s in result.plan.steps if isinstance(s, MarkForRevisionStep)]
    assert paths == ["Python > Async > Tasks", "Python > Async > Gather"]


def test_tool_call_parses_to_parsed_tool_call() -> None:
    """A TOOL_CALL block routes to the existing tool-call parser."""
    result = parse_plan_response(TOOL_CALL)

    assert isinstance(result, ParsedToolCall)
    assert result.calls[0].name == "get_weak_topics"


def test_teaching_leading_marker_rejected() -> None:
    """A TOPIC turn is not valid in the planner grammar.

    parse_response accepts TOPIC, parse_plan_response must not. This
    is what makes a wrong-kind terminal from the planner LLM die as a
    parse failure rather than being routed anywhere.
    """
    with pytest.raises(ParseError, match="planner response"):
        parse_plan_response("---TOPIC---\nx\n---END---\n")


def test_plan_without_end_raises() -> None:
    """A PLAN block missing its END marker raises."""
    text = '---PLAN---\n[{"tool": "mark_for_revision", "args": {"path": "A > B"}}]\n'
    with pytest.raises(ParseError, match="must be followed by END"):
        parse_plan_response(text)


def test_empty_plan_array_raises() -> None:
    """An empty array is a compliance failure, not an empty plan.

    The no_data guard runs before the planner chat opens, so weak
    topics always exist by the time the LLM plans. An empty array
    means the LLM misunderstood the format.
    """
    with pytest.raises(ParseError, match="must not be empty"):
        parse_plan_response("---PLAN---\n[]\n---END---\n")


def test_plan_body_must_be_array_not_object() -> None:
    """A single JSON object is rejected: the wire form is always an array."""
    text = '---PLAN---\n{"tool": "mark_for_revision", "args": {"path": "A > B"}}\n---END---\n'
    with pytest.raises(ParseError, match="must be a JSON array"):
        parse_plan_response(text)


def test_plan_with_non_object_entry_raises() -> None:
    """An array entry that is not an object names its index in the error."""
    with pytest.raises(ParseError, match="entry 0"):
        parse_plan_response('---PLAN---\n["not an object"]\n---END---\n')


def test_plan_with_unknown_tool_raises() -> None:
    """A step naming a tool outside the mutate vocabulary fails validation.

    mark_for_revision is the only mutate step. A get_weak_topics step
    (a read) or an invented tool name cannot validate against the
    mutate-step adapter, so a read step in a PLAN body dies here. That
    is what makes the plan mutate-only at the parse boundary.
    """
    with pytest.raises(ParseError, match="step 0 failed schema validation"):
        parse_plan_response('---PLAN---\n[{"tool": "get_weak_topics", "args": {}}]\n---END---\n')


def test_plan_step_missing_path_raises() -> None:
    """A mark_for_revision step without its required path fails validation."""
    with pytest.raises(ParseError, match="step 0 failed schema validation"):
        parse_plan_response('---PLAN---\n[{"tool": "mark_for_revision", "args": {}}]\n---END---\n')


def test_plan_body_invalid_json_raises() -> None:
    """A malformed JSON body raises with the JSON error surfaced."""
    with pytest.raises(ParseError, match="not valid JSON"):
        parse_plan_response("---PLAN---\n[{not json}]\n---END---\n")


def test_empty_text_raises() -> None:
    """No delimiters at all is a parse failure."""
    with pytest.raises(ParseError, match="No delimiters"):
        parse_plan_response("")
