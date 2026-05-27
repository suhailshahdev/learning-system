"""Tests for the teaching-turn evaluator and its aggregation.

The aggregation logic is a pure function tested directly with hand-fed
score lists: the falsifying tests feed scores that sit just over and just
under the variance ceiling and the pass threshold and assert the outcome
flips at the boundary, so a gate that ignored variance or threshold would
fail. The quorum tests feed fewer successes than quorum and assert ERROR.

The full evaluator is tested with two FakeTransports: one returning canned
teaching turns, one returning canned judge scores. Feeding the judge fake
a known score sequence drives the aggregation deterministically with no
real LLM.
"""

from __future__ import annotations

from app.eval.evaluators.teaching_eval import _aggregate, evaluate_teaching_item
from app.eval.schemas import ScoreOutcome, TeachingEvalItem, TeachingSetup
from app.models.enums import Difficulty, LearningMode
from app.transport.base import TransportError

from tests.services.fakes import FakeTransport

_JUDGE = "---SCORE---\n{score}\n---RATIONALE---\nok\n---END---"
_TEACHING_TURN = (
    "---TOPIC---\nPython > Basics\n---DIFFICULTY---\nbeginner\n---PREREQUISITES---\nNONE\n"
    "---MODE---\nflashcard\n---QUESTION---\nWhat does append do?\n---QUESTION_CODE---\nNONE\n"
    "---EXPECTED_ANSWER---\nAdds an item.\n---REQUIREMENTS---\nNONE\n---FOLLOWUP---\nNONE\n"
    "---TAGS---\nlists\n---END---"
)


def _item(pass_threshold: float = 0.6) -> TeachingEvalItem:
    return TeachingEvalItem(
        id="t",
        setup=TeachingSetup(
            topic_path="Python > Basics",
            mode=LearningMode.FLASHCARD,
            difficulty=Difficulty.BEGINNER,
        ),
        rubric="Is the question clear and at the right level?",
        pass_threshold=pass_threshold,
    )


# ---- aggregation (pure, hand-fed scores) ----


def test_aggregate_passes_above_threshold_low_variance() -> None:
    score = _aggregate(
        item_id="t",
        pass_threshold=0.6,
        n_runs=5,
        variance_ceiling=0.04,
        scores=[0.8, 0.8, 0.9, 0.8, 0.8],
        failures=[],
    )
    assert score.outcome == ScoreOutcome.PASS
    assert score.scores == [0.8, 0.8, 0.9, 0.8, 0.8]


def test_aggregate_fails_below_threshold() -> None:
    # Stable low scores: mean 0.3 < 0.6, variance ~0 -> FAIL not ERROR.
    score = _aggregate(
        item_id="t",
        pass_threshold=0.6,
        n_runs=5,
        variance_ceiling=0.04,
        scores=[0.3, 0.3, 0.3, 0.3, 0.3],
        failures=[],
    )
    assert score.outcome == ScoreOutcome.FAIL
    assert "below threshold" in score.detail


def test_aggregate_high_variance_is_error_not_pass() -> None:
    # Mean 0.66 clears 0.6, but scores swing 0.2..1.0. pvariance ~0.11 >
    # 0.04 -> ERROR. A gate ignoring variance would PASS this.
    score = _aggregate(
        item_id="t",
        pass_threshold=0.6,
        n_runs=5,
        variance_ceiling=0.04,
        scores=[0.2, 1.0, 0.9, 0.2, 1.0],
        failures=[],
    )
    assert score.outcome == ScoreOutcome.ERROR
    assert "variance" in score.detail


def test_aggregate_below_quorum_is_error() -> None:
    # 2 of 5 succeeded, quorum is 3 -> ERROR, math never runs on the 2.
    score = _aggregate(
        item_id="t",
        pass_threshold=0.6,
        n_runs=5,
        variance_ceiling=0.04,
        scores=[0.9, 0.9],
        failures=["boom", "boom", "boom"],
    )
    assert score.outcome == ScoreOutcome.ERROR
    assert "quorum" in score.detail


def test_aggregate_at_quorum_boundary_scores() -> None:
    # Exactly quorum (3 of 5) succeeded -> aggregation runs on the 3.
    score = _aggregate(
        item_id="t",
        pass_threshold=0.6,
        n_runs=5,
        variance_ceiling=0.04,
        scores=[0.8, 0.8, 0.8],
        failures=["boom", "boom"],
    )
    assert score.outcome == ScoreOutcome.PASS


# ---- full evaluator (two fakes) ----


async def test_evaluator_passes_with_consistent_judge() -> None:
    # 3 runs, each = one teaching turn + one judge score. Interleave the
    # canned responses in the order the evaluator consumes them per run:
    # teaching turn, then judge score.
    teaching = FakeTransport([_TEACHING_TURN, _TEACHING_TURN, _TEACHING_TURN])
    judge = FakeTransport(
        [_JUDGE.format(score="0.8"), _JUDGE.format(score="0.8"), _JUDGE.format(score="0.9")]
    )
    score = await evaluate_teaching_item(
        _item(pass_threshold=0.6),
        teaching_transport=teaching,
        judge_transport=judge,
        n_runs=3,
    )
    assert score.outcome == ScoreOutcome.PASS
    assert len(score.scores) == 3


async def test_evaluator_counts_driver_failures_toward_quorum() -> None:
    # Teaching transport fails every start, so every run fails at the driver
    # before the judge is consulted. 0 of 3 succeed -> ERROR below quorum.
    teaching = FakeTransport([], raise_on_send=TransportError("down"))
    judge = FakeTransport([_JUDGE.format(score="0.9")])
    score = await evaluate_teaching_item(
        _item(),
        teaching_transport=teaching,
        judge_transport=judge,
        n_runs=3,
    )
    assert score.outcome == ScoreOutcome.ERROR
    assert "0/3" in score.detail
