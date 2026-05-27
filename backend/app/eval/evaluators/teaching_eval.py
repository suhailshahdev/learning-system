"""LLM-as-judge evaluator for teaching-turn items.

Runs an item N times. Each run drives a teaching turn on the model under
test, then scores it on a separate judge model. The N scores are
aggregated into one ItemScore: PASS when enough runs succeed, the mean
clears the item's threshold, and the score variance is under the run's
ceiling. FAIL when the mean is too low or variance too high. ERROR when
too few runs even produced a judgeable turn.

Two transports, because the judge model must differ from the model under
test (a model grading its own output scores generously, enforced at
JudgeTarget construction). The teaching transport drives the turn, the
judge transport scores it.

A driver or judge failure on one run does not fail the item: transient
transport hiccups are tolerated as long as a majority of runs succeed.
Below that quorum the item scores ERROR (could not measure), distinct
from a FAIL (measured, judged poor). The variance gate treats an unstable
item, scores swinging across runs, as ERROR too: a wildly varying score
is not a reliable signal regardless of its mean.
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING, Any

from app.eval.judge import JudgeError, judge_teaching_turn
from app.eval.schemas import ItemScore, ScoreOutcome
from app.eval.teaching_driver import TeachingDriverError, drive_teaching_turn

if TYPE_CHECKING:
    from app.eval.schemas import TeachingEvalItem
    from app.transport.base import LLMTransport

# Default number of judged runs per item. Cost is low across the available
# transports, so the default favors a usable variance estimate over thrift.
# Overridable per run.
DEFAULT_N_RUNS = 5

# Default population-variance ceiling. Above this, an item's scores swing
# too much across runs to be a reliable signal and the item scores ERROR.
# First-guess value, tuning-pending against real judge behavior the way
# HANDOVER_THRESHOLD was tuned against real session data. 0.04 is roughly
# a 0.2 standard deviation on the 0.0-1.0-in-tenths score scale.
DEFAULT_VARIANCE_CEILING = 0.04


async def evaluate_teaching_item(
    item: TeachingEvalItem,
    *,
    teaching_transport: LLMTransport[Any],
    judge_transport: LLMTransport[Any],
    n_runs: int = DEFAULT_N_RUNS,
    variance_ceiling: float = DEFAULT_VARIANCE_CEILING,
) -> ItemScore:
    """Score one teaching item over n_runs and aggregate to an ItemScore.

    Each run drives a teaching turn and judges it. Successful run scores
    are aggregated, failed runs (driver or judge error) are counted and
    reported but do not by themselves fail the item unless they break
    quorum. Returns an ItemScore carrying the successful scores, the
    outcome, and a detail string with mean, variance, and run counts.
    """
    scores: list[float] = []
    failures: list[str] = []

    for _ in range(n_runs):
        try:
            turn = await drive_teaching_turn(teaching_transport, item.setup)
            score, _rationale = await judge_teaching_turn(
                judge_transport, item.setup, turn, item.rubric
            )
            scores.append(score)
        except (TeachingDriverError, JudgeError) as e:
            failures.append(e.message)

    return _aggregate(
        item_id=item.id,
        pass_threshold=item.pass_threshold,
        n_runs=n_runs,
        variance_ceiling=variance_ceiling,
        scores=scores,
        failures=failures,
    )


def _aggregate(
    *,
    item_id: str,
    pass_threshold: float,
    n_runs: int,
    variance_ceiling: float,
    scores: list[float],
    failures: list[str],
) -> ItemScore:
    """Turn N run results into one ItemScore.

    Order matters: the quorum check comes first, so mean and variance are
    only computed on a non-empty list. A majority of runs must have
    succeeded, below that the item is unmeasured and scores ERROR. With
    quorum met, an over-ceiling variance is ERROR (unstable signal), a
    mean below threshold is FAIL, and otherwise PASS.
    """
    succeeded = len(scores)
    quorum = n_runs // 2 + 1

    if succeeded < quorum:
        return ItemScore(
            item_id=item_id,
            outcome=ScoreOutcome.ERROR,
            detail=(
                f"only {succeeded}/{n_runs} runs succeeded (quorum {quorum}); failures: {failures}"
            ),
            scores=scores,
        )

    mean = statistics.mean(scores)
    variance = statistics.pvariance(scores)
    base = f"mean={mean:.3f} variance={variance:.4f} runs={succeeded}/{n_runs}"

    if variance > variance_ceiling:
        return ItemScore(
            item_id=item_id,
            outcome=ScoreOutcome.ERROR,
            detail=f"variance {variance:.4f} over ceiling {variance_ceiling} ({base})",
            scores=scores,
        )

    if mean < pass_threshold:
        return ItemScore(
            item_id=item_id,
            outcome=ScoreOutcome.FAIL,
            detail=f"mean {mean:.3f} below threshold {pass_threshold} ({base})",
            scores=scores,
        )

    return ItemScore(
        item_id=item_id,
        outcome=ScoreOutcome.PASS,
        detail=base,
        scores=scores,
    )
