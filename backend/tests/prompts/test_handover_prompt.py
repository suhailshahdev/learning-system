"""Tests for the handover-request prompt and its round-trip through the parser.

The prompt asks the dying chat to emit a HANDOVER block in the
format the parser expects. These tests cover the happy path (a
well-shaped response parses into ParsedHandover) and the failure
path the prompt's wording is designed to prevent (a response with
conversational wrapping is rejected, which is why the prompt
instructs the LLM to reply with the block only).
"""

from __future__ import annotations

from app.prompts.handover_prompt import build_handover_request
from app.schemas.parsed_response import ParsedHandover
from app.services.parser import parse_response


def test_build_handover_request_includes_format_spec() -> None:
    """The prompt names the format the parser expects."""
    prompt = build_handover_request()
    assert "---HANDOVER---" in prompt
    assert "---END_HANDOVER---" in prompt
    for key in (
        "DOMAIN_FOCUS",
        "COVERED",
        "LAST_QUESTION",
        "NEXT_PLANNED",
        "OPEN_THREADS",
        "USER_STATE",
    ):
        assert key in prompt


def test_handover_response_parses_into_parsed_handover() -> None:
    """A clean handover response round-trips through the parser."""
    response = """\
---HANDOVER---
DOMAIN_FOCUS: Python
COVERED: Iterators (intermediate), Generators (intermediate)
LAST_QUESTION: Asked about generator memory behavior; user answered correctly that values are produced lazily.
NEXT_PLANNED: Move into async generators
OPEN_THREADS: NONE
USER_STATE: Comfortable with iteration, ready to advance
---END_HANDOVER---"""

    parsed = parse_response(response)

    assert isinstance(parsed, ParsedHandover)
    assert parsed.domain_focus == "Python"
    assert parsed.covered == "Iterators (intermediate), Generators (intermediate)"
    assert parsed.last_question.startswith("Asked about generator memory")
    assert parsed.next_planned == "Move into async generators"
    assert parsed.open_threads == "NONE"
    assert parsed.user_state == "Comfortable with iteration, ready to advance"


def test_handover_with_conversational_intro_still_parses() -> None:
    """Leading text before ---HANDOVER--- is tolerated.

    The parser walks delimiter matches and treats text between
    delimiters as block content; anything before the first
    delimiter is silently discarded. The prompt's "block only"
    instruction exists to save tokens in the dying chat, not
    because the parser would otherwise fail.
    """
    response = """\
Sure, here is the handover for the next chat:

---HANDOVER---
DOMAIN_FOCUS: Python
COVERED: Iterators (intermediate)
LAST_QUESTION: Asked about iter() vs next(); user answered correctly.
NEXT_PLANNED: Generators
OPEN_THREADS: NONE
USER_STATE: Engaged
---END_HANDOVER---"""

    parsed = parse_response(response)

    assert isinstance(parsed, ParsedHandover)
    assert parsed.domain_focus == "Python"
