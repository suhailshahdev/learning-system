"""Tests for the eval runner's dispatch and record assembly.

Parser dispatch runs with no transport and produces a record with a parser
target. Teaching dispatch runs with two FakeTransports through a
TeachingRunContext and produces a record with a judge target. The
no-context guard asserts a teaching set run without a context raises
EvalRunnerError before any item runs.
"""

from __future__ import annotations

import pytest
from app.eval.runner import EvalRunnerError, TeachingRunContext, run_set
from app.eval.schemas import (
    EvalKind,
    ParserEvalItem,
    ParserEvalSet,
    ParsesTo,
    ScoreOutcome,
    TeachingEvalItem,
    TeachingEvalSet,
    TeachingSetup,
)
from app.eval.targets import JudgeTarget, ParserTarget
from app.models.enums import Difficulty, LearningMode

from tests.services.fakes import FakeTransport

_TEACHING_TURN = (
    "---TOPIC---\nPython > Basics\n---DIFFICULTY---\nbeginner\n---PREREQUISITES---\nNONE\n"
    "---MODE---\nflashcard\n---QUESTION---\nWhat does append do?\n---QUESTION_CODE---\nNONE\n"
    "---EXPECTED_ANSWER---\nAdds an item.\n---REQUIREMENTS---\nNONE\n---FOLLOWUP---\nNONE\n"
    "---TAGS---\nlists\n---END---"
)
_JUDGE = "---SCORE---\n0.8\n---RATIONALE---\nok\n---END---"


def _parser_set() -> ParserEvalSet:
    return ParserEvalSet(
        name="p",
        schema_version=1,
        items=[
            ParserEvalItem(
                id="p1",
                raw="not a delimited response",
                expected=ParsesTo(kind="turn"),
            )
        ],
    )


def _teaching_set() -> TeachingEvalSet:
    return TeachingEvalSet(
        name="t",
        schema_version=1,
        items=[
            TeachingEvalItem(
                id="t1",
                setup=TeachingSetup(
                    topic_path="Python > Basics",
                    mode=LearningMode.FLASHCARD,
                    difficulty=Difficulty.BEGINNER,
                ),
                rubric="clear?",
                pass_threshold=0.6,
            )
        ],
    )


async def test_run_parser_set_produces_parser_target() -> None:
    record = await run_set(_parser_set(), "hash-p")
    assert isinstance(record.target, ParserTarget)
    assert record.eval_kind == EvalKind.PARSER
    # The one parser item expects a turn but the raw does not parse, so it
    # fails: this asserts the runner actually invoked the evaluator.
    assert record.scores[0].outcome == ScoreOutcome.FAIL
    assert record.set_content_hash == "hash-p"


async def test_run_teaching_set_produces_judge_target() -> None:
    teaching = FakeTransport([_TEACHING_TURN, _TEACHING_TURN, _TEACHING_TURN])
    judge = FakeTransport([_JUDGE, _JUDGE, _JUDGE])
    context = TeachingRunContext(
        teaching_transport=teaching,
        judge_transport=judge,
        transport="deepseek",
        model_under_test="deepseek-chat",
        judge_model="claude-playwright",
        n_runs=3,
    )
    record = await run_set(_teaching_set(), "hash-t", teaching_context=context)
    assert isinstance(record.target, JudgeTarget)
    assert record.target.judge_model == "claude-playwright"
    assert record.scores[0].outcome == ScoreOutcome.PASS
    assert len(record.scores[0].scores) == 3


async def test_teaching_set_without_context_raises() -> None:
    with pytest.raises(EvalRunnerError, match="requires a teaching_context"):
        await run_set(_teaching_set(), "hash-t")
