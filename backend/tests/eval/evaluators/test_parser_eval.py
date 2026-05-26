"""Tests for the parser-robustness evaluator.

The evaluator runs parse_response on an item's raw string and scores the
outcome against the item's expectation. These tests cover the four
outcome combinations (expected-parse/got-parse, expected-parse/got-raise,
expected-raise/got-parse, expected-raise/got-raise), field-assertion
matching and mismatching, the message-substring check, and the boundary
where a sentinel-cleared field is asserted via the string "None".

The combinations are designed to falsify a broken evaluator: an evaluator
that let ParseError escape would crash test_expected_raise_got_raise. One
that ignored fields would pass test_field_mismatch_fails. One that ignored
the message substring would pass test_raise_wrong_message_fails.
"""

from __future__ import annotations

from app.eval.evaluators.parser_eval import evaluate_parser_item
from app.eval.schemas import ParserEvalItem, ScoreOutcome

# A clean teaching turn in wire format. Reused across tests that need a
# parse to succeed. Field order matches the parser's expected sequence.
_CLEAN_TURN = (
    "---TOPIC---\n"
    "Python > Basics > Lists\n"
    "---DIFFICULTY---\n"
    "beginner\n"
    "---PREREQUISITES---\n"
    "NONE\n"
    "---MODE---\n"
    "flashcard\n"
    "---QUESTION---\n"
    "What does append do?\n"
    "---QUESTION_CODE---\n"
    "NONE\n"
    "---EXPECTED_ANSWER---\n"
    "Adds an item to the end of a list.\n"
    "---REQUIREMENTS---\n"
    "NONE\n"
    "---FOLLOWUP---\n"
    "NONE\n"
    "---TAGS---\n"
    "lists\n"
    "---END---"
)


def _item(raw: str, expected: dict[str, object]) -> ParserEvalItem:
    """Build a ParserEvalItem from a raw string and an expected dict."""
    return ParserEvalItem.model_validate({"id": "t", "raw": raw, "expected": expected})


def test_expected_parse_got_parse_passes() -> None:
    item = _item(_CLEAN_TURN, {"outcome": "parses_to", "kind": "turn"})
    score = evaluate_parser_item(item)
    assert score.outcome == ScoreOutcome.PASS


def test_field_assertion_matches() -> None:
    item = _item(
        _CLEAN_TURN,
        {
            "outcome": "parses_to",
            "kind": "turn",
            "fields": {"topic_path": "Python > Basics > Lists", "difficulty": "beginner"},
        },
    )
    assert evaluate_parser_item(item).outcome == ScoreOutcome.PASS


def test_field_mismatch_fails() -> None:
    item = _item(
        _CLEAN_TURN,
        {"outcome": "parses_to", "kind": "turn", "fields": {"difficulty": "advanced"}},
    )
    score = evaluate_parser_item(item)
    assert score.outcome == ScoreOutcome.FAIL
    assert "difficulty" in score.detail


def test_sentinel_cleared_field_asserted_as_none_string() -> None:
    # REQUIREMENTS: NONE parses to requirements=None, which dumps to JSON
    # null and stringifies to "None". An item can assert that via "None".
    item = _item(
        _CLEAN_TURN,
        {"outcome": "parses_to", "kind": "turn", "fields": {"requirements": "None"}},
    )
    assert evaluate_parser_item(item).outcome == ScoreOutcome.PASS


def test_wrong_kind_fails() -> None:
    item = _item(_CLEAN_TURN, {"outcome": "parses_to", "kind": "session_end"})
    score = evaluate_parser_item(item)
    assert score.outcome == ScoreOutcome.FAIL
    assert "session_end" in score.detail


def test_expected_parse_got_raise_fails() -> None:
    item = _item("not a delimited response at all", {"outcome": "parses_to", "kind": "turn"})
    score = evaluate_parser_item(item)
    assert score.outcome == ScoreOutcome.FAIL


def test_expected_raise_got_raise_passes() -> None:
    item = _item("not a delimited response at all", {"outcome": "raises"})
    assert evaluate_parser_item(item).outcome == ScoreOutcome.PASS


def test_expected_raise_got_parse_fails() -> None:
    item = _item(_CLEAN_TURN, {"outcome": "raises"})
    score = evaluate_parser_item(item)
    assert score.outcome == ScoreOutcome.FAIL


def test_raise_message_substring_matches() -> None:
    # A turn missing fields raises with a message naming the field count.
    truncated = "---TOPIC---\nPython > Basics\n---END---"
    item = _item(truncated, {"outcome": "raises", "message_contains": "fields"})
    assert evaluate_parser_item(item).outcome == ScoreOutcome.PASS


def test_raise_wrong_message_fails() -> None:
    truncated = "---TOPIC---\nPython > Basics\n---END---"
    item = _item(truncated, {"outcome": "raises", "message_contains": "completely unrelated text"})
    score = evaluate_parser_item(item)
    assert score.outcome == ScoreOutcome.FAIL
    assert "does not contain" in score.detail
