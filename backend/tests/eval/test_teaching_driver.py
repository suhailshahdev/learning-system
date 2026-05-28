"""Tests for the teaching-turn driver.

The driver opens a fresh chat with a static intro and parses the response
into a teaching turn. These tests use FakeTransport with canned responses:
the happy path returns a real teaching-turn wire string and asserts it
parses to a ParsedTurn through the real parser (which catches format-spec
drift in the static intro). The failure paths assert TeachingDriverError
on transport failure, on an unparseable response, and on a response that
parses to a non-teaching kind.
"""

from __future__ import annotations

from app.eval.schemas import TeachingSetup
from app.eval.teaching_driver import (
    TeachingDriverError,
    _build_eval_first_prompt,
    drive_teaching_turn,
)
from app.models.enums import Difficulty, LearningMode
from app.transport.base import TransportError

from tests.services.fakes import FakeTransport

_TEACHING_TURN_WIRE = (
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

_SESSION_END_WIRE = "---SESSION_END_PROPOSAL---\nAll done.\n---END---"


def _setup() -> TeachingSetup:
    return TeachingSetup(
        topic_path="Python > Basics > Lists",
        mode=LearningMode.FLASHCARD,
        difficulty=Difficulty.BEGINNER,
    )


async def test_drives_teaching_turn() -> None:
    transport = FakeTransport([_TEACHING_TURN_WIRE])
    parsed = await drive_teaching_turn(transport, _setup())
    assert parsed.kind == "turn"
    assert parsed.topic_path == "Python > Basics > Lists"
    assert parsed.mode == LearningMode.FLASHCARD


def test_eval_prompt_pins_mode_and_difficulty() -> None:
    # The prompt must name the setup's mode and difficulty: the rubric is
    # written for them, and the smoke showed the teacher drifts to
    # flashcard/beginner without an explicit instruction not to.
    setup = TeachingSetup(
        topic_path="System Design > Caching > Cache Invalidation",
        mode=LearningMode.EXPLAIN_BACK,
        difficulty=Difficulty.INTERMEDIATE,
    )
    prompt = _build_eval_first_prompt(setup)
    assert "explain_back" in prompt
    assert "intermediate" in prompt
    assert "System Design > Caching > Cache Invalidation" in prompt


async def test_transport_failure_raises_driver_error() -> None:
    transport = FakeTransport([], raise_on_send=TransportError("boom"))
    try:
        await drive_teaching_turn(transport, _setup())
    except TeachingDriverError as e:
        assert "Transport failed" in e.message
    else:
        raise AssertionError("expected TeachingDriverError")


async def test_unparseable_response_raises_driver_error() -> None:
    transport = FakeTransport(["this is not a delimited response"])
    try:
        await drive_teaching_turn(transport, _setup())
    except TeachingDriverError as e:
        assert "did not parse" in e.message
    else:
        raise AssertionError("expected TeachingDriverError")


async def test_non_teaching_kind_raises_driver_error() -> None:
    transport = FakeTransport([_SESSION_END_WIRE])
    try:
        await drive_teaching_turn(transport, _setup())
    except TeachingDriverError as e:
        assert "session_end" in e.message
    else:
        raise AssertionError("expected TeachingDriverError")
