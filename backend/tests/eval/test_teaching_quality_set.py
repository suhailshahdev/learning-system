"""Structural test for the shipped teaching-quality eval set.

Unlike the parser set, this set cannot be load-and-run tested: running it
calls real LLMs and is nondeterministic, so a "every item passes" assertion
would be neither free nor stable. The deterministic surface is structural:
the set loads, validates as a TeachingEvalSet, has unique ids, and every
pass_threshold is in range. Whether a real teacher and judge produce sane
scores is a smoke step, gathered as evidence, not asserted here (LLM
behavior is verified by evidence, not strict assertion).
"""

from __future__ import annotations

from pathlib import Path

from app.eval.loader import load_set
from app.eval.schemas import TeachingEvalSet

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_SET_PATH = _BACKEND_ROOT / "eval" / "sets" / "teaching_turn_quality.json"


def test_teaching_set_loads_and_validates() -> None:
    loaded = load_set(_SET_PATH)
    assert isinstance(loaded, TeachingEvalSet)
    assert len(loaded.items) >= 1


def test_teaching_set_ids_unique() -> None:
    loaded = load_set(_SET_PATH)
    ids = [item.id for item in loaded.items]
    assert len(ids) == len(set(ids))


def test_teaching_set_thresholds_in_range() -> None:
    loaded = load_set(_SET_PATH)
    assert isinstance(loaded, TeachingEvalSet)
    for item in loaded.items:
        assert 0.0 <= item.pass_threshold <= 1.0
