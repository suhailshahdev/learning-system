"""Tests for the LLM-as-judge client and judge-response parser.

The parser tests cover well-formed extraction, leading-prose tolerance,
each missing-field case, empty rationale, and the score-validation
boundary (non-numeric, out-of-range, and the rounding of float-parse
artifacts into a valid bucket). The client test drives a FakeTransport
returning a canned judge response and asserts the score and rationale
come through, the transport-failure test asserts JudgeError.

Whether a real LLM follows the judge intro is a smoke-level question, not
covered here: these tests prove the parser handles judge output, not that
a real judge produces compliant output.
"""

from __future__ import annotations

import pytest
from app.eval.judge import JudgeError, judge_teaching_turn, parse_judge_response
from app.eval.schemas import TeachingSetup
from app.models.enums import Difficulty, LearningMode
from app.schemas.parsed_response import ParsedTurn
from app.transport.base import TransportError

from tests.services.fakes import FakeTransport

_WELL_FORMED = "---SCORE---\n0.8\n---RATIONALE---\nClear question at the right level.\n---END---"


def _setup() -> TeachingSetup:
    return TeachingSetup(
        topic_path="Python > Basics > Lists",
        mode=LearningMode.FLASHCARD,
        difficulty=Difficulty.BEGINNER,
    )


def _turn() -> ParsedTurn:
    return ParsedTurn(
        topic_path="Python > Basics > Lists",
        difficulty=Difficulty.BEGINNER,
        mode=LearningMode.FLASHCARD,
        question="What does append do?",
        expected_answer="Adds an item to the end of a list.",
    )


def test_parse_well_formed() -> None:
    score, rationale = parse_judge_response(_WELL_FORMED)
    assert score == 0.8
    assert rationale == "Clear question at the right level."


def test_parse_tolerates_leading_prose() -> None:
    text = "Here is my assessment:\n\n" + _WELL_FORMED
    score, _ = parse_judge_response(text)
    assert score == 0.8


def test_parse_missing_score_raises() -> None:
    text = "---RATIONALE---\nSome reasoning.\n---END---"
    with pytest.raises(JudgeError, match="missing SCORE"):
        parse_judge_response(text)


def test_parse_missing_rationale_raises() -> None:
    text = "---SCORE---\n0.5\n---END---"
    with pytest.raises(JudgeError, match="missing RATIONALE"):
        parse_judge_response(text)


def test_parse_empty_rationale_raises() -> None:
    text = "---SCORE---\n0.5\n---RATIONALE---\n\n---END---"
    with pytest.raises(JudgeError, match="empty RATIONALE"):
        parse_judge_response(text)


def test_parse_non_numeric_score_raises() -> None:
    text = "---SCORE---\nhigh\n---RATIONALE---\nGood.\n---END---"
    with pytest.raises(JudgeError, match="not a number"):
        parse_judge_response(text)


def test_parse_out_of_range_score_raises() -> None:
    text = "---SCORE---\n1.5\n---RATIONALE---\nToo generous.\n---END---"
    with pytest.raises(JudgeError, match="not one of the quantized"):
        parse_judge_response(text)


def test_parse_score_rounds_float_artifact() -> None:
    # A judge emitting 0.30 parses to 0.3 and validates. A near-bucket
    # float artifact rounds into the bucket rather than failing.
    text = "---SCORE---\n0.30\n---RATIONALE---\nFine.\n---END---"
    score, _ = parse_judge_response(text)
    assert score == 0.3


async def test_judge_drives_transport() -> None:
    transport = FakeTransport([_WELL_FORMED])
    score, rationale = await judge_teaching_turn(transport, _setup(), _turn(), "Is it clear?")
    assert score == 0.8
    assert rationale == "Clear question at the right level."


async def test_judge_transport_failure_raises() -> None:
    transport = FakeTransport([], raise_on_send=TransportError("down"))
    with pytest.raises(JudgeError, match="transport failed"):
        await judge_teaching_turn(transport, _setup(), _turn(), "Is it clear?")
