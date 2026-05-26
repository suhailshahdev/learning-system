"""Consistency test for the shipped parser-robustness eval set.

Loads backend/eval/sets/parser_robustness.json, runs every item through
the evaluator, and asserts each one PASSES. This is distinct from the
evaluator's own unit tests, which use hand-built items: this test asserts
the shipped set is internally consistent, that every item's expectation
matches what the parser actually does. An authoring error in the set (a
wrong message fragment, a wire string that does not parse to the claimed
kind) fails here rather than only surfacing on a manual eval run.

The set-file path is resolved relative to the backend root so the test
runs from any working directory. Coupling the test to the committed set's
location is intentional: the set is a shipped artifact and a test that
its items are self-consistent should break loudly if someone edits it
wrong.
"""

from __future__ import annotations

from pathlib import Path

from app.eval.evaluators.parser_eval import evaluate_parser_item
from app.eval.loader import load_set
from app.eval.schemas import ParserEvalSet, ScoreOutcome

# backend/tests/eval/test_parser_robustness_set.py -> backend/
_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_SET_PATH = _BACKEND_ROOT / "eval" / "sets" / "parser_robustness.json"


def test_parser_robustness_set_all_items_pass() -> None:
    loaded = load_set(_SET_PATH)
    assert isinstance(loaded, ParserEvalSet)

    failures: list[str] = []
    for item in loaded.items:
        score = evaluate_parser_item(item)
        if score.outcome != ScoreOutcome.PASS:
            failures.append(f"{item.id}: {score.outcome.value} - {score.detail}")

    assert not failures, "set items did not all pass:\n" + "\n".join(failures)


def test_parser_robustness_set_ids_unique() -> None:
    loaded = load_set(_SET_PATH)
    ids = [item.id for item in loaded.items]
    assert len(ids) == len(set(ids)), "duplicate item ids in set"
