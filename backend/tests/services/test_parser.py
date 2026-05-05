"""Tests for app.services.parser.

Cases are inline strings rather than separate fixture files. The
test cases double as documentation of the wire format. Each test
covers one specific shape (happy path) or one specific failure
mode (error path).
"""

from __future__ import annotations

import pytest
from app.models.enums import Difficulty, GradingVerdict, LearningMode
from app.schemas.parsed_response import ParsedHandover, ParsedSessionEnd, ParsedTurn
from app.services.parser import ParseError, parse_response

# Happy-path turn with all fields populated meaningfully. Represents
# a mid-session follow-up turn (grading present, real verdict and
# explanation).
TURN_FULL = """\
---TOPIC---
Python > Data Types > Integers
---DIFFICULTY---
beginner
---PREREQUISITES---
Python > Basics:beginner, Python > Variables:beginner
---MODE---
flashcard
---GRADING---
correct
---GRADING_EXPLANATION---
Right. Floor division rounds toward negative infinity, so 7 // 2 is 3.
---QUESTION---
What is the result of 7 // 2 in Python 3?
---EXPECTED_ANSWER---
3
---REQUIREMENTS---
Python 3.12+
---FOLLOWUP---
Why does floor division round toward negative infinity?
---TAGS---
arithmetic, integers
---END---
"""

# Happy-path turn with all sentinels in play. Represents the first
# turn of a session: grading is NONE (no previous answer), expected
# answer is OPEN, requirements and followup are NONE.
TURN_SENTINELS = """\
---TOPIC---
Python > Concepts > Decorators
---DIFFICULTY---
intermediate
---PREREQUISITES---
NONE
---MODE---
socratic
---GRADING---
NONE
---GRADING_EXPLANATION---
NONE
---QUESTION---
Walk me through what a decorator does conceptually.
---EXPECTED_ANSWER---
OPEN
---REQUIREMENTS---
NONE
---FOLLOWUP---
NONE
---TAGS---

---END---
"""

# A session-end proposal.
SESSION_END = """\
---SESSION_END_PROPOSAL---
You've covered Python integer arithmetic. Ready to mark these as learned.
---END---
"""

# A handover block.
HANDOVER = """\
---HANDOVER---
DOMAIN_FOCUS: Python
COVERED: Integers (beginner), Floats (beginner)
LAST_QUESTION: What is 7 // 2?
NEXT_PLANNED: Boolean operations
OPEN_THREADS: User asked about complex numbers
USER_STATE: Confident on basic arithmetic
---END_HANDOVER---
"""


class TestParseTurn:
    def test_full_turn_parses(self) -> None:
        result = parse_response(TURN_FULL)
        assert isinstance(result, ParsedTurn)
        assert result.kind == "turn"
        assert result.topic_path == "Python > Data Types > Integers"
        assert result.difficulty == Difficulty.BEGINNER
        assert result.mode == LearningMode.FLASHCARD
        assert result.grading_verdict == GradingVerdict.CORRECT
        assert result.grading_explanation is not None
        assert "Floor division" in result.grading_explanation
        assert result.question == "What is the result of 7 // 2 in Python 3?"
        assert result.expected_answer == "3"
        assert result.requirements == "Python 3.12+"
        assert result.followup == "Why does floor division round toward negative infinity?"
        assert result.tags == ["arithmetic", "integers"]
        assert len(result.prerequisites) == 2
        assert result.prerequisites[0].topic_path == "Python > Basics"
        assert result.prerequisites[0].min_difficulty == Difficulty.BEGINNER

    def test_sentinels_become_none_or_empty(self) -> None:
        result = parse_response(TURN_SENTINELS)
        assert isinstance(result, ParsedTurn)
        assert result.prerequisites == []
        assert result.grading_verdict is None
        assert result.grading_explanation is None
        assert result.expected_answer is None
        assert result.requirements is None
        assert result.followup is None
        assert result.tags == []


class TestParseSessionEnd:
    def test_session_end_parses(self) -> None:
        result = parse_response(SESSION_END)
        assert isinstance(result, ParsedSessionEnd)
        assert result.kind == "session_end"
        assert "Python integer arithmetic" in result.summary


class TestParseHandover:
    def test_handover_parses(self) -> None:
        result = parse_response(HANDOVER)
        assert isinstance(result, ParsedHandover)
        assert result.kind == "handover"
        assert result.domain_focus == "Python"
        assert result.covered == "Integers (beginner), Floats (beginner)"
        assert result.last_question == "What is 7 // 2?"
        assert result.next_planned == "Boolean operations"
        assert result.open_threads == "User asked about complex numbers"
        assert result.user_state == "Confident on basic arithmetic"


class TestParseErrors:
    def test_empty_input_raises(self) -> None:
        with pytest.raises(ParseError, match="No delimiters"):
            parse_response("")

    def test_unknown_leading_delimiter_raises(self) -> None:
        with pytest.raises(ParseError, match="Unknown leading delimiter"):
            parse_response("---NONSENSE---\nbody\n---END---\n")

    def test_turn_missing_end_marker_raises(self) -> None:
        text = TURN_FULL.replace("---END---\n", "")
        with pytest.raises(ParseError, match="terminate with ---END---"):
            parse_response(text)

    def test_turn_wrong_field_order_raises(self) -> None:
        # Swap DIFFICULTY and TOPIC blocks
        text = TURN_FULL.replace(
            "---TOPIC---\nPython > Data Types > Integers\n---DIFFICULTY---\nbeginner",
            "---DIFFICULTY---\nbeginner\n---TOPIC---\nPython > Data Types > Integers",
        )
        with pytest.raises(ParseError):
            parse_response(text)

    def test_invalid_difficulty_raises(self) -> None:
        text = TURN_FULL.replace("beginner", "expert")
        with pytest.raises(ParseError, match="Invalid DIFFICULTY"):
            parse_response(text)

    def test_invalid_mode_raises(self) -> None:
        text = TURN_FULL.replace("flashcard", "telepathy")
        with pytest.raises(ParseError, match="Invalid MODE"):
            parse_response(text)

    def test_invalid_grading_verdict_raises(self) -> None:
        text = TURN_FULL.replace("---GRADING---\ncorrect", "---GRADING---\nbrilliant")
        with pytest.raises(ParseError, match="Invalid GRADING"):
            parse_response(text)

    def test_grading_field_missing_raises(self) -> None:
        # Drop both grading blocks entirely - the parser should detect
        # the missing fields rather than silently allowing the older format.
        text = TURN_FULL.replace(
            "---GRADING---\ncorrect\n---GRADING_EXPLANATION---\n"
            "Right. Floor division rounds toward negative infinity, so 7 // 2 is 3.\n",
            "",
        )
        with pytest.raises(ParseError):
            parse_response(text)

    def test_malformed_prerequisite_raises(self) -> None:
        text = TURN_FULL.replace(
            "Python > Basics:beginner, Python > Variables:beginner",
            "Python > Basics, Python > Variables:beginner",
        )
        with pytest.raises(ParseError, match="Malformed prerequisite"):
            parse_response(text)

    def test_handover_missing_field_raises(self) -> None:
        text = HANDOVER.replace("USER_STATE: Confident on basic arithmetic\n", "")
        with pytest.raises(ParseError, match="missing fields"):
            parse_response(text)

    def test_handover_unknown_key_raises(self) -> None:
        text = HANDOVER.replace("DOMAIN_FOCUS:", "MYSTERY_KEY:")
        with pytest.raises(ParseError, match="Unknown handover key"):
            parse_response(text)
