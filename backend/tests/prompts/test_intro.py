"""Tests for app.prompts.intro.

The intro is a static string but produced from enum values, so the
tests verify (a) every enum value the format references actually
appears in the rendered string and (b) the structural delimiters
match what the parser expects. Tests catch silent drift between
'what we tell the LLM' and 'what we accept.'
"""

from __future__ import annotations

import pytest
from app.models.enums import Difficulty, GradingVerdict, LearningMode
from app.prompts.intro import build_intro


@pytest.fixture(scope="module")
def intro() -> str:
    """Render the intro once; tests just inspect it."""
    return build_intro()


class TestIntroContent:
    def test_every_learning_mode_appears(self, intro: str) -> None:
        for mode in LearningMode:
            assert mode.value in intro, f"{mode.value!r} missing from intro"

    def test_every_difficulty_appears(self, intro: str) -> None:
        for diff in Difficulty:
            assert diff.value in intro, f"{diff.value!r} missing from intro"

    def test_every_grading_verdict_appears(self, intro: str) -> None:
        for verdict in GradingVerdict:
            assert verdict.value in intro, f"{verdict.value!r} missing from intro"

    def test_required_section_headers_present(self, intro: str) -> None:
        for header in (
            "OUTPUT FORMAT",
            "LEARNING MODES",
            "GRADING VERDICTS",
            "DIFFICULTY VALUES",
            "RULES",
        ):
            assert header in intro

    def test_turn_format_delimiters_present(self, intro: str) -> None:
        for delimiter in (
            "---TOPIC---",
            "---DIFFICULTY---",
            "---PREREQUISITES---",
            "---MODE---",
            "---GRADING---",
            "---GRADING_EXPLANATION---",
            "---GRADING_EXPLANATION_CODE---",
            "---QUESTION---",
            "---QUESTION_CODE---",
            "---EXPECTED_ANSWER---",
            "---REQUIREMENTS---",
            "---FOLLOWUP---",
            "---TAGS---",
            "---END---",
        ):
            assert delimiter in intro

    def test_session_end_delimiter_present(self, intro: str) -> None:
        assert "---SESSION_END_PROPOSAL---" in intro

    def test_handover_delimiters_present(self, intro: str) -> None:
        assert "---HANDOVER---" in intro
        assert "---END_HANDOVER---" in intro
        for key in (
            "DOMAIN_FOCUS",
            "COVERED",
            "LAST_QUESTION",
            "NEXT_PLANNED",
            "OPEN_THREADS",
            "USER_STATE",
        ):
            assert key in intro

    def test_sentinel_words_documented(self, intro: str) -> None:
        # OPEN and NONE both have specific meanings in the format
        # spec; the parser converts them to None. The intro must
        # tell the LLM about both.
        assert "OPEN" in intro
        assert "NONE" in intro


class TestIntroIntegrity:
    def test_intro_is_non_empty(self, intro: str) -> None:
        assert intro.strip(), "build_intro() returned empty string"

    def test_intro_is_str(self, intro: str) -> None:
        assert isinstance(intro, str)

    def test_intro_starts_without_leading_blank_line(self, intro: str) -> None:
        # Triple-quoted string with `\` escape should not produce
        # a leading newline; a regression here would push the first
        # real line off the LLM's attention.
        assert not intro.startswith("\n")
