"""Tests for the regression report differ.

Each test traces a known input pair through build_report and asserts the
specific, located result, not just that a diff exists. The regression test
flips one item pass->fail and asserts that item id appears in regressions:
a differ that noticed a change but misattributed it would fail. The error-
isolation test flips pass->error and asserts the item lands in other_flips,
not regressions, so a broken run is never read as a behavior regression.
"""

from __future__ import annotations

from datetime import UTC, datetime

from app.eval.report import build_report
from app.eval.schemas import EvalKind, ItemScore, RunRecord, ScoreOutcome
from app.eval.targets import ParserTarget


def _run(scores: list[ItemScore], content_hash: str = "h1") -> RunRecord:
    """Build a parser run record with the given scores."""
    now = datetime(2026, 5, 26, 12, 0, 0, tzinfo=UTC)
    return RunRecord(
        run_id="r",
        set_name="parser_robustness",
        set_content_hash=content_hash,
        eval_kind=EvalKind.PARSER,
        target=ParserTarget(),
        started_at=now,
        finished_at=now,
        scores=scores,
    )


def _score(item_id: str, outcome: ScoreOutcome) -> ItemScore:
    return ItemScore(item_id=item_id, outcome=outcome)


def test_first_run_has_no_previous() -> None:
    # Trace: previous_run=None -> early return, has_previous False, delta None.
    this = _run([_score("a", ScoreOutcome.PASS)])
    report = build_report(this, None)
    assert report.has_previous is False
    assert report.previous_pass_rate is None
    assert report.pass_rate_delta is None
    assert report.regressions == []


def test_identical_runs_no_flips() -> None:
    # Trace: same ids, same outcomes -> intersection has no before!=after.
    scores = [_score("a", ScoreOutcome.PASS), _score("b", ScoreOutcome.PASS)]
    report = build_report(_run(scores), _run(scores))
    assert report.regressions == []
    assert report.fixes == []
    assert report.other_flips == []
    assert report.pass_rate_delta == 0.0


def test_regression_located_by_item_id() -> None:
    # Trace: prev a=PASS, this a=FAIL. Intersection {a}, before PASS != after
    # FAIL, to_outcome FAIL and from != FAIL -> regressions. b unchanged.
    prev = _run([_score("a", ScoreOutcome.PASS), _score("b", ScoreOutcome.PASS)])
    this = _run([_score("a", ScoreOutcome.FAIL), _score("b", ScoreOutcome.PASS)])
    report = build_report(this, prev)

    assert [f.item_id for f in report.regressions] == ["a"]
    assert report.regressions[0].from_outcome == ScoreOutcome.PASS
    assert report.regressions[0].to_outcome == ScoreOutcome.FAIL
    assert report.fixes == []
    # Pass rate fell from 1.0 to 0.5.
    assert report.pass_rate_delta == -0.5


def test_fix_located_by_item_id() -> None:
    # Trace: prev a=FAIL, this a=PASS. to_outcome PASS, from != PASS -> fixes.
    prev = _run([_score("a", ScoreOutcome.FAIL)])
    this = _run([_score("a", ScoreOutcome.PASS)])
    report = build_report(this, prev)

    assert [f.item_id for f in report.fixes] == ["a"]
    assert report.regressions == []
    assert report.pass_rate_delta == 1.0


def test_error_transition_is_not_a_regression() -> None:
    # Trace: prev a=PASS, this a=ERROR. to_outcome ERROR (not FAIL, not PASS)
    # -> other_flips, NOT regressions. A broken run must not read as a
    # behavior regression.
    prev = _run([_score("a", ScoreOutcome.PASS)])
    this = _run([_score("a", ScoreOutcome.ERROR)])
    report = build_report(this, prev)

    assert report.regressions == []
    assert [f.item_id for f in report.other_flips] == ["a"]
    assert report.other_flips[0].to_outcome == ScoreOutcome.ERROR


def test_added_and_removed_items() -> None:
    # Trace: prev {a, b}, this {a, c}. Intersection {a} unchanged. c only in
    # this -> added. b only in prev -> removed. Neither is a flip.
    prev = _run([_score("a", ScoreOutcome.PASS), _score("b", ScoreOutcome.PASS)])
    this = _run([_score("a", ScoreOutcome.PASS), _score("c", ScoreOutcome.PASS)])
    report = build_report(this, prev)

    assert report.added_item_ids == ["c"]
    assert report.removed_item_ids == ["b"]
    assert report.regressions == []
    assert report.fixes == []


def test_set_hash_change_flagged() -> None:
    # Trace: different content hashes -> set_hash_changed True. Delta still
    # computed but the flag warns a reader the denominators may differ.
    prev = _run([_score("a", ScoreOutcome.PASS)], content_hash="old")
    this = _run([_score("a", ScoreOutcome.PASS)], content_hash="new")
    report = build_report(this, prev)
    assert report.set_hash_changed is True


def test_multiple_flips_each_classified() -> None:
    # Trace: a PASS->FAIL (regression), b FAIL->PASS (fix), c PASS->ERROR
    # (other). All three in intersection, each classified by direction.
    prev = _run(
        [
            _score("a", ScoreOutcome.PASS),
            _score("b", ScoreOutcome.FAIL),
            _score("c", ScoreOutcome.PASS),
        ]
    )
    this = _run(
        [
            _score("a", ScoreOutcome.FAIL),
            _score("b", ScoreOutcome.PASS),
            _score("c", ScoreOutcome.ERROR),
        ]
    )
    report = build_report(this, prev)

    assert [f.item_id for f in report.regressions] == ["a"]
    assert [f.item_id for f in report.fixes] == ["b"]
    assert [f.item_id for f in report.other_flips] == ["c"]
