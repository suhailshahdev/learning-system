"""Run an eval set against its evaluator and produce a run record.

The runner loads a set, scores every item through the evaluator that
matches the set's kind, and assembles a RunRecord. Writing the record and
diffing against the previous run are separate steps (run_record.write_run
and the regression report) so the runner's job is purely produce-the-record.

Evaluator dispatch is by eval_kind. Only the parser evaluator exists now,
retrieval and teaching raise NotImplementedError until their sub-phases
land. The dispatch seam is built for all three from the start so adding an
evaluator is filling a branch, not reshaping the runner.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from app.eval.evaluators.parser_eval import evaluate_parser_item
from app.eval.evaluators.teaching_eval import (
    DEFAULT_N_RUNS,
    DEFAULT_VARIANCE_CEILING,
    evaluate_teaching_item,
)
from app.eval.schemas import (
    EvalKind,
    ItemScore,
    ParserEvalSet,
    RunRecord,
    TeachingEvalSet,
)
from app.eval.targets import JudgeTarget, ParserTarget

if TYPE_CHECKING:
    from app.eval.schemas import EvalSet
    from app.eval.targets import TargetDescriptor
    from app.transport.base import LLMTransport


class EvalRunnerError(Exception):
    """A run could not proceed for a reason the runner detects up front.

    Distinct from a NotImplementedError (an unbuilt evaluator) and from an
    item scoring ERROR (a per-item measurement failure). Raised when a set
    is run without the context its kind requires, before any item runs.
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@dataclass(frozen=True)
class TeachingRunContext:
    """Live wiring a teaching set needs to run.

    The two transports are runtime objects used during the run, not the
    serialized TargetDescriptor written to the record: one drives the turn
    on the model under test, the other judges it on a different model.
    model_under_test and judge_model are the names recorded on the run's
    JudgeTarget, transport is the recorded transport kind string. n_runs
    and variance_ceiling tune the run and default to the evaluator's
    module constants.
    """

    teaching_transport: LLMTransport[Any]
    judge_transport: LLMTransport[Any]
    transport: str
    model_under_test: str
    judge_model: str
    n_runs: int = DEFAULT_N_RUNS
    variance_ceiling: float = DEFAULT_VARIANCE_CEILING


async def run_set(
    eval_set: EvalSet,
    set_content_hash: str,
    *,
    teaching_context: TeachingRunContext | None = None,
) -> RunRecord:
    """Score every item in a set and return a run record.

    set_content_hash is passed in rather than computed here so the caller
    controls when hashing happens (the CLI hashes once at load and reuses
    the value). teaching_context is required for a teaching set and ignored
    for other kinds. A teaching set run without it raises EvalRunnerError
    before any item runs. started_at and finished_at bracket the scoring.
    """
    started_at = datetime.now(UTC)
    scores, target = await _score_set(eval_set, teaching_context)
    finished_at = datetime.now(UTC)

    return RunRecord(
        run_id=str(uuid.uuid4()),
        set_name=eval_set.name,
        set_content_hash=set_content_hash,
        eval_kind=eval_set.eval_kind,
        target=target,
        started_at=started_at,
        finished_at=finished_at,
        scores=scores,
        cost_usd=0.0,
    )


async def _score_set(
    eval_set: EvalSet, teaching_context: TeachingRunContext | None
) -> tuple[list[ItemScore], TargetDescriptor]:
    """Dispatch to the evaluator for the set's kind, returning scores and target.

    Returns the target alongside the scores because the target is
    kind-specific: a parser run has no model, a teaching run names a
    transport and judge. Bundling them keeps run_set's record assembly
    uniform.
    """
    if isinstance(eval_set, ParserEvalSet):
        scores = [evaluate_parser_item(item) for item in eval_set.items]
        return scores, ParserTarget()

    if eval_set.eval_kind == EvalKind.RETRIEVAL:
        msg = "retrieval evaluator is not implemented yet"
        raise NotImplementedError(msg)

    # eval_set is narrowed to TeachingEvalSet here: not ParserEvalSet (the
    # isinstance above) and not retrieval (the check above). mypy proves
    # this is the only remaining member.
    return await _score_teaching_set(eval_set, teaching_context)


async def _score_teaching_set(
    eval_set: TeachingEvalSet, teaching_context: TeachingRunContext | None
) -> tuple[list[ItemScore], TargetDescriptor]:
    """Score a teaching set, requiring the live transport context.

    Builds the JudgeTarget for the record from the context's model names.
    Its constructor enforces judge_model != model_under_test, so a config
    that pairs a model with itself fails here before any LLM call.
    """
    if teaching_context is None:
        msg = "teaching set requires a teaching_context (two transports); none was given"
        raise EvalRunnerError(msg)

    target = JudgeTarget(
        transport=teaching_context.transport,
        model_under_test=teaching_context.model_under_test,
        judge_model=teaching_context.judge_model,
    )

    scores = [
        await evaluate_teaching_item(
            item,
            teaching_transport=teaching_context.teaching_transport,
            judge_transport=teaching_context.judge_transport,
            n_runs=teaching_context.n_runs,
            variance_ceiling=teaching_context.variance_ceiling,
        )
        for item in eval_set.items
    ]
    return scores, target
