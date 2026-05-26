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
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from app.eval.evaluators.parser_eval import evaluate_parser_item
from app.eval.schemas import (
    EvalKind,
    ItemScore,
    ParserEvalSet,
    RunRecord,
)
from app.eval.targets import ParserTarget

if TYPE_CHECKING:
    from app.eval.schemas import EvalSet
    from app.eval.targets import TargetDescriptor


def run_set(eval_set: EvalSet, set_content_hash: str) -> RunRecord:
    """Score every item in a set and return a run record.

    set_content_hash is passed in rather than computed here so the caller
    controls when hashing happens (the CLI hashes once at load and reuses
    the value). started_at and finished_at bracket the scoring loop.
    """
    started_at = datetime.now(UTC)
    scores, target = _score_set(eval_set)
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


def _score_set(eval_set: EvalSet) -> tuple[list[ItemScore], TargetDescriptor]:
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
        msg = "retrieval evaluator is not implemented yet (lands in M12.5)"
        raise NotImplementedError(msg)

    # eval_set is narrowed to TeachingEvalSet here: not ParserEvalSet (the
    # isinstance above) and not retrieval (the check above). mypy proves
    # this is the only remaining member, so it is the unconditional tail
    # with no fallback after it. Adding a fourth EvalKind would make mypy
    # flag a missing return, forcing the new branch rather than letting it
    # fall through at runtime.
    msg = "teaching evaluator is not implemented yet (lands in M12.4)"
    raise NotImplementedError(msg)
