"""Tests for app.prompts.intro.

The intro is produced from enum values plus dynamic database
state (existing domains, user knowledge summary). Tests verify
(a) every enum value the format references appears in the
rendered string, (b) the structural delimiters match what the
parser expects, and (c) the dynamic sections render correctly
for both empty and populated databases.

Tests catch silent drift between 'what we tell the LLM' and
'what we accept.'
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from app.models import Domain, UserKnowledgeAssertion
from app.models.enums import (
    AssertionSource,
    Difficulty,
    DomainKind,
    GradingVerdict,
    LearningMode,
)
from app.prompts.intro import build_intro

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession


@pytest.fixture
async def empty_intro(db: DbSession) -> str:
    """Render the intro against an empty database.

    Function-scoped because the underlying db fixture is
    function-scoped (each test gets a fresh in-memory DB).
    """
    return await build_intro(db)


@pytest.fixture
async def populated_intro(db: DbSession) -> str:
    """Render the intro after seeding domains and assertions.

    Verifies the dynamic sections render with real data. Adds
    two domains and three knowledge assertions covering two of
    them at different difficulties.
    """
    db.add(Domain(name="Python", kind=DomainKind.LANGUAGE, description="Python language"))
    db.add(Domain(name="FastAPI", kind=DomainKind.FRAMEWORK, description=None))
    db.add(
        UserKnowledgeAssertion(
            topic_path="Python > Data Types > Integers",
            difficulty=Difficulty.INTERMEDIATE,
            source=AssertionSource.SELF_DECLARED,
        )
    )
    db.add(
        UserKnowledgeAssertion(
            topic_path="Python > Data Types > Strings",
            difficulty=Difficulty.INTERMEDIATE,
            source=AssertionSource.SELF_DECLARED,
        )
    )
    db.add(
        UserKnowledgeAssertion(
            topic_path="FastAPI > Routing > Path Parameters",
            difficulty=Difficulty.BEGINNER,
            source=AssertionSource.DERIVED_FROM_LEARNED_ITEMS,
        )
    )
    db.commit()
    return await build_intro(db)


class TestIntroContent:
    async def test_every_learning_mode_appears(self, empty_intro: str) -> None:
        for mode in LearningMode:
            assert mode.value in empty_intro, f"{mode.value!r} missing from intro"

    async def test_every_difficulty_appears(self, empty_intro: str) -> None:
        for diff in Difficulty:
            assert diff.value in empty_intro, f"{diff.value!r} missing from intro"

    async def test_every_grading_verdict_appears(self, empty_intro: str) -> None:
        for verdict in GradingVerdict:
            assert verdict.value in empty_intro, f"{verdict.value!r} missing from intro"

    async def test_required_section_headers_present(self, empty_intro: str) -> None:
        for header in (
            "TURN FLOW",
            "OUTPUT FORMAT",
            "LEARNING MODES",
            "GRADING VERDICTS",
            "DIFFICULTY VALUES",
            "EXISTING DOMAINS",
            "USER KNOWLEDGE",
            "AVAILABLE TOOLS",
            "RULES",
        ):
            assert header in empty_intro

    async def test_teaching_turn_delimiters_present(self, empty_intro: str) -> None:
        # The teaching turn block, post-split, has these fields. The
        # GRADING fields moved to the standalone grading response.
        for delimiter in (
            "---TOPIC---",
            "---DIFFICULTY---",
            "---PREREQUISITES---",
            "---MODE---",
            "---QUESTION---",
            "---QUESTION_CODE---",
            "---EXPECTED_ANSWER---",
            "---REQUIREMENTS---",
            "---FOLLOWUP---",
            "---TAGS---",
            "---END---",
        ):
            assert delimiter in empty_intro

    async def test_grading_response_delimiters_present(self, empty_intro: str) -> None:
        # The standalone grading response block, post-split.
        for delimiter in (
            "---GRADING---",
            "---GRADING_EXPLANATION---",
            "---GRADING_EXPLANATION_CODE---",
            "---END---",
        ):
            assert delimiter in empty_intro

    async def test_session_end_delimiter_present(self, empty_intro: str) -> None:
        assert "---SESSION_END_PROPOSAL---" in empty_intro

    async def test_handover_delimiters_present(self, empty_intro: str) -> None:
        assert "---HANDOVER---" in empty_intro
        assert "---END_HANDOVER---" in empty_intro
        for key in (
            "DOMAIN_FOCUS",
            "COVERED",
            "LAST_QUESTION",
            "NEXT_PLANNED",
            "OPEN_THREADS",
            "USER_STATE",
        ):
            assert key in empty_intro

    async def test_tool_call_delimiter_present(self, empty_intro: str) -> None:
        # The four reactive tools are invoked via this delimiter,
        # parser will key off it.
        assert "---TOOL_CALL---" in empty_intro

    async def test_advertised_tools_listed(self, empty_intro: str) -> None:
        # Only the four reactive tools should be advertised. The two
        # tools whose data is pre-loaded (list_domains,
        # get_user_knowledge_summary) exist in the registry but are
        # not advertised so the LLM doesn't waste turns calling them.
        for tool_name in (
            "get_topics_by_domain",
            "create_domain",
            "create_or_update_topic",
            "get_recent_sessions",
        ):
            assert tool_name in empty_intro

    async def test_unadvertised_tools_not_listed(self, empty_intro: str) -> None:
        # These handlers exist but their data is in the intro
        # already, they should not appear in the AVAILABLE TOOLS
        # list. They may appear elsewhere in the intro (e.g. in
        # comments) but not as a callable advertised to the LLM.
        # A cheap way to check this is to find the AVAILABLE TOOLS
        # section and verify the names aren't in it.
        tools_start = empty_intro.find("AVAILABLE TOOLS")
        assert tools_start >= 0
        rules_start = empty_intro.find("RULES")
        assert rules_start > tools_start
        tools_section = empty_intro[tools_start:rules_start]
        assert "list_domains" not in tools_section
        assert "get_user_knowledge_summary" not in tools_section

    async def test_sentinel_words_documented(self, empty_intro: str) -> None:
        # OPEN and NONE both have specific meanings in the format
        # spec, the parser converts them to None. The intro must
        # tell the LLM about both.
        assert "OPEN" in empty_intro
        assert "NONE" in empty_intro


class TestEmptyState:
    """Sections handle the no-data case with a clear marker."""

    async def test_empty_domains_message(self, empty_intro: str) -> None:
        assert "(none yet" in empty_intro

    async def test_empty_knowledge_message(self, empty_intro: str) -> None:
        assert "(no prior knowledge recorded" in empty_intro


class TestPopulatedState:
    """Sections render real data when the database has rows."""

    async def test_domains_appear_with_kind(self, populated_intro: str) -> None:
        assert "Python (language)" in populated_intro
        assert "FastAPI (framework)" in populated_intro

    async def test_domain_description_included_when_set(self, populated_intro: str) -> None:
        assert "Python language" in populated_intro

    async def test_knowledge_grouped_by_domain(self, populated_intro: str) -> None:
        # Python should have an intermediate (2) entry from the two
        # asserted topics; FastAPI should have a beginner (1) entry.
        assert "Python:" in populated_intro
        assert "intermediate (2)" in populated_intro
        assert "FastAPI:" in populated_intro
        assert "beginner (1)" in populated_intro


class TestIntroIntegrity:
    async def test_intro_is_non_empty(self, empty_intro: str) -> None:
        assert empty_intro.strip(), "build_intro() returned empty string"

    async def test_intro_is_str(self, empty_intro: str) -> None:
        assert isinstance(empty_intro, str)

    async def test_intro_starts_without_leading_blank_line(self, empty_intro: str) -> None:
        # Triple-quoted string with `\` escape should not produce
        # a leading newline,  a regression here would push the first
        # real line off the LLM's attention.
        assert not empty_intro.startswith("\n")
