"""Diff one eval run against the previous run of the same set.

A run record is a snapshot. The regression report turns two snapshots into
a verdict: did the pass rate move, which items flipped outcome, did the set
itself change between runs. The report is a data structure, rendering it to
text is a separate formatter, so the diff logic is testable without
capturing output.

Flipped items are computed over the intersection of item ids present in
both runs: an item only in one run has no before-and-after to compare, so
it is reported as added or removed, not as a regression. Pass rate is each
run's own pass fraction over its own item count. When the set content hash
differs between runs, the denominators differ too, and the hash-changed
flag tells a reader not to over-read the rate delta.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.eval.schemas import ScoreOutcome

if TYPE_CHECKING:
    from app.eval.schemas import ItemScore, RunRecord


@dataclass(frozen=True)
class ItemFlip:
    """One item that changed outcome between the previous run and this one."""

    item_id: str
    from_outcome: ScoreOutcome
    to_outcome: ScoreOutcome


@dataclass(frozen=True)
class RegressionReport:
    """The diff between this run and the previous run of the same set.

    previous is None on a set's first-ever run: there is nothing to diff
    against, so the report carries this run's pass rate and empty flip
    lists. regressions are pass-or-error to fail flips, fixes are fail-or-
    error to pass flips, other_flips covers transitions that are neither
    clearly a regression nor a fix (pass to error, error to fail). added
    and removed are item ids present in only one run.
    """

    set_name: str
    has_previous: bool
    this_pass_rate: float
    previous_pass_rate: float | None
    pass_rate_delta: float | None
    set_hash_changed: bool
    regressions: list[ItemFlip] = field(default_factory=list)
    fixes: list[ItemFlip] = field(default_factory=list)
    other_flips: list[ItemFlip] = field(default_factory=list)
    added_item_ids: list[str] = field(default_factory=list)
    removed_item_ids: list[str] = field(default_factory=list)


def build_report(this_run: RunRecord, previous_run: RunRecord | None) -> RegressionReport:
    """Diff this run against the previous run of the same set.

    previous_run is None when this is the set's first run. The two runs are
    assumed to share a set name, the caller pairs them (read_last_run keys
    on set name).
    """
    this_rate = _pass_rate(this_run.scores)

    if previous_run is None:
        return RegressionReport(
            set_name=this_run.set_name,
            has_previous=False,
            this_pass_rate=this_rate,
            previous_pass_rate=None,
            pass_rate_delta=None,
            set_hash_changed=False,
        )

    prev_rate = _pass_rate(previous_run.scores)
    prev_by_id = {s.item_id: s for s in previous_run.scores}
    this_by_id = {s.item_id: s for s in this_run.scores}

    regressions: list[ItemFlip] = []
    fixes: list[ItemFlip] = []
    other_flips: list[ItemFlip] = []

    for item_id in prev_by_id.keys() & this_by_id.keys():
        before = prev_by_id[item_id].outcome
        after = this_by_id[item_id].outcome
        if before == after:
            continue
        flip = ItemFlip(item_id=item_id, from_outcome=before, to_outcome=after)
        _classify_flip(flip, regressions, fixes, other_flips)

    return RegressionReport(
        set_name=this_run.set_name,
        has_previous=True,
        this_pass_rate=this_rate,
        previous_pass_rate=prev_rate,
        pass_rate_delta=this_rate - prev_rate,
        set_hash_changed=this_run.set_content_hash != previous_run.set_content_hash,
        regressions=sorted(regressions, key=lambda f: f.item_id),
        fixes=sorted(fixes, key=lambda f: f.item_id),
        other_flips=sorted(other_flips, key=lambda f: f.item_id),
        added_item_ids=sorted(this_by_id.keys() - prev_by_id.keys()),
        removed_item_ids=sorted(prev_by_id.keys() - this_by_id.keys()),
    )


def _classify_flip(
    flip: ItemFlip,
    regressions: list[ItemFlip],
    fixes: list[ItemFlip],
    other_flips: list[ItemFlip],
) -> None:
    """Sort a flip into regression, fix, or other by its direction."""
    if flip.to_outcome == ScoreOutcome.FAIL and flip.from_outcome != ScoreOutcome.FAIL:
        regressions.append(flip)
    elif flip.to_outcome == ScoreOutcome.PASS and flip.from_outcome != ScoreOutcome.PASS:
        fixes.append(flip)
    else:
        other_flips.append(flip)


def _pass_rate(scores: list[ItemScore]) -> float:
    """Fraction of scores that passed. Zero for an empty score list."""
    if not scores:
        return 0.0
    passed = sum(1 for s in scores if s.outcome == ScoreOutcome.PASS)
    return passed / len(scores)
