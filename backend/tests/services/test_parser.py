"""Tests for app.services.parser.

Cases are inline strings rather than separate fixture files. The
test cases double as documentation of the wire format. Each test
covers one specific shape (happy path) or one specific failure
mode (error path).
"""

from __future__ import annotations

import pytest
from app.models.enums import Difficulty, DomainKind, GradingVerdict, LearningMode
from app.schemas.parsed_response import (
    ParsedGrading,
    ParsedHandover,
    ParsedProposal,
    ParsedSessionEnd,
    ParsedToolCall,
    ParsedTurn,
)
from app.schemas.tools import (
    CreateDomainCall,
    CreateOrUpdateTopicCall,
    GetTopicsByDomainCall,
)
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
---QUESTION---
What is the result of 7 // 2 in Python 3?
---QUESTION_CODE---
NONE
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
---QUESTION---
Walk me through what a decorator does conceptually.
---QUESTION_CODE---
NONE
---EXPECTED_ANSWER---
OPEN
---REQUIREMENTS---
NONE
---FOLLOWUP---
NONE
---TAGS---

---END---
"""

# Happy-path turn with code blocks in QUESTION and GRADING_EXPLANATION.
# Represents the "show me the output of this code" pattern that hits
# every code-bearing question in real sessions.
TURN_WITH_CODE = """\
---TOPIC---
Python > Control Flow > For Loops
---DIFFICULTY---
beginner
---PREREQUISITES---
NONE
---MODE---
type_the_answer
---QUESTION---
What does this script print? One number per line.
---QUESTION_CODE---
python
for i in range(3):
    print(i * 2)
---EXPECTED_ANSWER---
0
2
4
---REQUIREMENTS---
NONE
---FOLLOWUP---
NONE
---TAGS---
python, loops, range
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


# Standalone grading response with no code block.
GRADING_SIMPLE = """\
---GRADING---
correct
---GRADING_EXPLANATION---
Right. Floor division rounds toward negative infinity, so 7 // 2 is 3.
---GRADING_EXPLANATION_CODE---
NONE
---END---
"""

# Standalone grading response with an explanation code block.
GRADING_WITH_CODE = """\
---GRADING---
partial
---GRADING_EXPLANATION---
Close, but the loop body multiplies by 2, not adds 2. Trace it again
with i = 0, 1, 2.
---GRADING_EXPLANATION_CODE---
python
print(0 * 2)  # 0
print(1 * 2)  # 2
print(2 * 2)  # 4
---END---
"""

# Standalone grading response for a free-form mode (explain_back,
# socratic). Verdict is open_graded, explanation carries the
# teaching feedback.
GRADING_OPEN_GRADED = """\
---GRADING---
open_graded
---GRADING_EXPLANATION---
You're on the right track with the decorator pattern, but you missed
that closures keep a reference to the enclosing scope. Try writing one
without functools.wraps and see what happens to the wrapped function's
name.
---GRADING_EXPLANATION_CODE---
NONE
---END---
"""


# Tool calls. The block body is a JSON object with `name` and `args`
# keys. Pydantic's discriminated union narrows to the specific call
# variant based on `name`.

TOOL_CALL_LIST_DOMAINS = """\
---TOOL_CALL---
{"name": "list_domains", "args": {}}
---END---
"""

TOOL_CALL_GET_TOPICS = """\
---TOOL_CALL---
{"name": "get_topics_by_domain", "args": {"domain_name": "Python"}}
---END---
"""

TOOL_CALL_CREATE_DOMAIN = """\
---TOOL_CALL---
{"name": "create_domain", "args": {"name": "Rust", "kind": "language", "description": "Systems language"}}
---END---
"""

TOOL_CALL_CREATE_OR_UPDATE_TOPIC = """\
---TOOL_CALL---
{"name": "create_or_update_topic", "args": {"path": "Python > Data Types > Integers", "difficulty": "beginner", "prerequisites": [{"topic_path": "Python > Basics", "min_difficulty": "beginner"}]}}
---END---
"""

# Happy-path proposal block. Two required keys, terminated by
# END_PROPOSAL. Mirrors the handover format shape.
PROPOSAL_HAPPY = """\
---PROPOSAL---
TOPIC_PATH: Python > Data Types > Integers
REASONING: You have 4 incorrect attempts on integer arithmetic in the last week. Worth revisiting before moving to floats.
---END_PROPOSAL---
"""


class TestParseTurn:
    def test_full_turn_parses(self) -> None:
        result = parse_response(TURN_FULL)
        assert isinstance(result, ParsedTurn)
        assert result.kind == "turn"
        assert result.topic_path == "Python > Data Types > Integers"
        assert result.difficulty == Difficulty.BEGINNER
        assert result.mode == LearningMode.FLASHCARD
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
        assert result.expected_answer is None
        assert result.requirements is None
        assert result.followup is None
        assert result.question_code is None
        assert result.tags == []

    def test_code_block_parses_into_code_block_model(self) -> None:
        result = parse_response(TURN_WITH_CODE)
        assert isinstance(result, ParsedTurn)

        assert result.question_code is not None
        assert result.question_code.language == "python"
        assert result.question_code.body == "for i in range(3):\n    print(i * 2)"

    def test_code_block_with_none_sentinel_becomes_none(self) -> None:
        # TURN_FULL has its code block set to NONE.
        result = parse_response(TURN_FULL)
        assert isinstance(result, ParsedTurn)
        assert result.question_code is None


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


class TestParseProposal:
    def test_proposal_parses(self) -> None:
        result = parse_response(PROPOSAL_HAPPY)
        assert isinstance(result, ParsedProposal)
        assert result.kind == "proposal"
        assert result.topic_path == "Python > Data Types > Integers"
        assert "incorrect attempts" in result.reasoning


class TestParseGrading:
    def test_simple_grading_parses(self) -> None:
        result = parse_response(GRADING_SIMPLE)
        assert isinstance(result, ParsedGrading)
        assert result.kind == "grading"
        assert result.verdict == GradingVerdict.CORRECT
        assert "Floor division" in result.explanation
        assert result.explanation_code is None

    def test_grading_with_code_block_parses(self) -> None:
        result = parse_response(GRADING_WITH_CODE)
        assert isinstance(result, ParsedGrading)
        assert result.verdict == GradingVerdict.PARTIAL
        assert "Trace it again" in result.explanation
        assert result.explanation_code is not None
        assert result.explanation_code.language == "python"
        assert "print(0 * 2)" in result.explanation_code.body

    def test_open_graded_verdict_parses(self) -> None:
        result = parse_response(GRADING_OPEN_GRADED)
        assert isinstance(result, ParsedGrading)
        assert result.verdict == GradingVerdict.OPEN_GRADED
        assert "decorator pattern" in result.explanation
        assert result.explanation_code is None


class TestParseToolCall:
    def test_no_args_tool_call_parses(self) -> None:
        result = parse_response(TOOL_CALL_LIST_DOMAINS)
        assert isinstance(result, ParsedToolCall)
        assert result.kind == "tool_call"
        assert result.calls[0].name == "list_domains"
        # raw_text preserves the original block content for error_log.
        assert "list_domains" in result.raw_text

    def test_simple_args_tool_call_parses(self) -> None:
        result = parse_response(TOOL_CALL_GET_TOPICS)
        assert isinstance(result, ParsedToolCall)
        assert result.calls[0].name == "get_topics_by_domain"
        assert isinstance(result.calls[0], GetTopicsByDomainCall)
        assert result.calls[0].args.domain_name == "Python"

    def test_create_domain_tool_call_parses(self) -> None:
        result = parse_response(TOOL_CALL_CREATE_DOMAIN)
        assert isinstance(result, ParsedToolCall)
        assert isinstance(result.calls[0], CreateDomainCall)
        assert result.calls[0].args.name == "Rust"
        assert result.calls[0].args.kind == DomainKind.LANGUAGE
        assert result.calls[0].args.description == "Systems language"

    def test_create_or_update_topic_tool_call_parses(self) -> None:
        result = parse_response(TOOL_CALL_CREATE_OR_UPDATE_TOPIC)
        assert isinstance(result, ParsedToolCall)
        assert isinstance(result.calls[0], CreateOrUpdateTopicCall)
        assert result.calls[0].args.path == "Python > Data Types > Integers"
        assert result.calls[0].args.difficulty == Difficulty.BEGINNER
        assert len(result.calls[0].args.prerequisites) == 1
        assert result.calls[0].args.prerequisites[0].topic_path == "Python > Basics"
        assert result.calls[0].args.prerequisites[0].min_difficulty == Difficulty.BEGINNER


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

    def test_code_block_missing_body_raises(self) -> None:
        # Language tag on first line, no body after.
        text = TURN_WITH_CODE.replace(
            "---QUESTION_CODE---\npython\nfor i in range(3):\n    print(i * 2)\n",
            "---QUESTION_CODE---\npython\n",
        )
        with pytest.raises(ParseError, match="QUESTION_CODE"):
            parse_response(text)

    def test_code_block_missing_language_raises(self) -> None:
        # Empty first line, body on second line.
        text = TURN_WITH_CODE.replace(
            "---QUESTION_CODE---\npython\nfor i in range(3):\n    print(i * 2)\n",
            "---QUESTION_CODE---\n\nfor i in range(3):\n",
        )
        with pytest.raises(ParseError, match="QUESTION_CODE"):
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

    def test_proposal_missing_end_marker_raises(self) -> None:
        text = PROPOSAL_HAPPY.replace("---END_PROPOSAL---\n", "")
        with pytest.raises(ParseError, match="PROPOSAL must be followed by END_PROPOSAL"):
            parse_response(text)

    def test_proposal_missing_topic_path_raises(self) -> None:
        text = PROPOSAL_HAPPY.replace("TOPIC_PATH: Python > Data Types > Integers\n", "")
        with pytest.raises(ParseError, match="missing fields"):
            parse_response(text)

    def test_proposal_missing_reasoning_raises(self) -> None:
        text = PROPOSAL_HAPPY.replace(
            "REASONING: You have 4 incorrect attempts on integer "
            "arithmetic in the last week. Worth revisiting before "
            "moving to floats.\n",
            "",
        )
        with pytest.raises(ParseError, match="missing fields"):
            parse_response(text)

    def test_proposal_unknown_key_raises(self) -> None:
        text = PROPOSAL_HAPPY.replace("TOPIC_PATH:", "MYSTERY_KEY:")
        with pytest.raises(ParseError, match="Unknown proposal key"):
            parse_response(text)

    def test_proposal_malformed_line_raises(self) -> None:
        text = PROPOSAL_HAPPY.replace(
            "TOPIC_PATH: Python > Data Types > Integers",
            "TOPIC_PATH without a colon",
        )
        with pytest.raises(ParseError, match="Malformed proposal line"):
            parse_response(text)

    def test_proposal_empty_topic_path_raises(self) -> None:
        text = PROPOSAL_HAPPY.replace(
            "TOPIC_PATH: Python > Data Types > Integers",
            "TOPIC_PATH: ",
        )
        with pytest.raises(ParseError, match="schema validation"):
            parse_response(text)

    def test_proposal_empty_reasoning_raises(self) -> None:
        text = PROPOSAL_HAPPY.replace(
            "REASONING: You have 4 incorrect attempts on integer "
            "arithmetic in the last week. Worth revisiting before "
            "moving to floats.",
            "REASONING: ",
        )
        with pytest.raises(ParseError, match="schema validation"):
            parse_response(text)

    def test_tool_call_missing_end_raises(self) -> None:
        text = TOOL_CALL_LIST_DOMAINS.replace("---END---\n", "")
        with pytest.raises(ParseError, match="TOOL_CALL must be followed by END"):
            parse_response(text)

    def test_tool_call_invalid_json_raises(self) -> None:
        text = "---TOOL_CALL---\nthis is not json\n---END---\n"
        with pytest.raises(ParseError, match="not valid JSON"):
            parse_response(text)

    def test_tool_call_non_object_json_raises(self) -> None:
        text = "---TOOL_CALL---\n[1, 2, 3]\n---END---\n"
        with pytest.raises(ParseError, match="must be a JSON object"):
            parse_response(text)

    def test_tool_call_unknown_tool_name_raises(self) -> None:
        text = '---TOOL_CALL---\n{"name": "fly_to_moon", "args": {}}\n---END---\n'
        with pytest.raises(ParseError, match="schema validation"):
            parse_response(text)

    def test_tool_call_invalid_args_raises(self) -> None:
        # create_domain requires `name` and `kind`. Missing required fields.
        text = '---TOOL_CALL---\n{"name": "create_domain", "args": {}}\n---END---\n'
        with pytest.raises(ParseError, match="schema validation"):
            parse_response(text)

    def test_tool_call_extra_args_raises(self) -> None:
        # tools.py uses extra="forbid" so unknown args fail validation.
        text = (
            "---TOOL_CALL---\n"
            '{"name": "list_domains", "args": {"unexpected": "value"}}\n'
            "---END---\n"
        )
        with pytest.raises(ParseError, match="schema validation"):
            parse_response(text)

    def test_tool_call_empty_body_raises(self) -> None:
        text = "---TOOL_CALL---\n\n---END---\n"
        with pytest.raises(ParseError, match="empty"):
            parse_response(text)

    def test_grading_missing_end_marker_raises(self) -> None:
        text = GRADING_SIMPLE.replace("---END---\n", "")
        with pytest.raises(ParseError, match="terminate with ---END---"):
            parse_response(text)

    def test_grading_invalid_verdict_raises(self) -> None:
        text = GRADING_SIMPLE.replace("correct", "magnificent")
        with pytest.raises(ParseError, match="Invalid GRADING"):
            parse_response(text)

    def test_grading_missing_explanation_raises(self) -> None:
        # Drop the GRADING_EXPLANATION block entirely. The parser should
        # detect the missing field rather than silently accepting two
        # fields where three are required.
        text = GRADING_SIMPLE.replace(
            "---GRADING_EXPLANATION---\n"
            "Right. Floor division rounds toward negative infinity, so 7 // 2 is 3.\n",
            "",
        )
        with pytest.raises(ParseError):
            parse_response(text)

    def test_grading_empty_explanation_raises(self) -> None:
        text = GRADING_SIMPLE.replace(
            "---GRADING_EXPLANATION---\n"
            "Right. Floor division rounds toward negative infinity, so 7 // 2 is 3.\n",
            "---GRADING_EXPLANATION---\n\n",
        )
        with pytest.raises(ParseError, match="schema validation"):
            parse_response(text)
