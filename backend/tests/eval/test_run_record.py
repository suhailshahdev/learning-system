"""Tests for the run-record writer and previous-run lookup.

Writing covers the round-trip (write then read back equal) and the
filename shape (timestamp prefix, set-name slug). Reading covers the
empty cases (absent dir, no match), latest-by-started_at selection,
exact set-name matching, and skipping a malformed file without losing
a valid one.

The latest-selection test deliberately writes records whose start times
do not match their write order, so an implementation that trusted write
order or filename sort instead of the started_at field would fail it.
The set-name test uses parser vs parser_robustness, the exact pair a
substring match gets wrong.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from app.eval.run_record import read_last_run, write_run
from app.eval.schemas import EvalKind, RunRecord
from app.eval.targets import ParserTarget

if TYPE_CHECKING:
    from pathlib import Path


def _record(set_name: str, started_at: datetime) -> RunRecord:
    """A minimal valid parser run record with the given name and start time."""
    return RunRecord(
        run_id=f"run-{started_at.strftime('%Y%m%d%H%M%S')}",
        set_name=set_name,
        set_content_hash="abc123",
        eval_kind=EvalKind.PARSER,
        target=ParserTarget(),
        started_at=started_at,
        finished_at=started_at,
        scores=[],
    )


def test_write_then_read_round_trips(tmp_path: Path) -> None:
    record = _record("parser_smoke", datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC))
    write_run(record, tmp_path)

    result = read_last_run("parser_smoke", tmp_path)
    assert result is not None
    assert result.run_id == record.run_id
    assert result.set_name == "parser_smoke"


def test_write_filename_has_timestamp_and_name(tmp_path: Path) -> None:
    record = _record("parser_smoke", datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC))
    path = write_run(record, tmp_path)
    assert path.name == "20260525T120000-parser_smoke.json"


def test_write_creates_missing_dir(tmp_path: Path) -> None:
    nested = tmp_path / "runs" / "nested"
    record = _record("parser_smoke", datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC))
    write_run(record, nested)
    assert nested.is_dir()


def test_read_absent_dir_returns_none(tmp_path: Path) -> None:
    assert read_last_run("anything", tmp_path / "does_not_exist") is None


def test_read_no_match_returns_none(tmp_path: Path) -> None:
    write_run(_record("other_set", datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC)), tmp_path)
    assert read_last_run("parser_smoke", tmp_path) is None


def test_read_returns_latest_by_started_at(tmp_path: Path) -> None:
    # Write the older record last, so write order is the reverse of time
    # order. An implementation trusting write order would pick this one.
    newer = _record("s", datetime(2026, 5, 25, 15, 0, 0, tzinfo=UTC))
    older = _record("s", datetime(2026, 5, 25, 9, 0, 0, tzinfo=UTC))
    write_run(newer, tmp_path)
    write_run(older, tmp_path)

    result = read_last_run("s", tmp_path)
    assert result is not None
    assert result.started_at == newer.started_at


def test_read_exact_set_name_not_substring(tmp_path: Path) -> None:
    write_run(_record("parser_robustness", datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC)), tmp_path)
    # "parser" is a prefix of "parser_robustness". A substring match would
    # wrongly return the robustness record here.
    assert read_last_run("parser", tmp_path) is None


def test_read_skips_malformed_file(tmp_path: Path) -> None:
    valid = _record("s", datetime(2026, 5, 25, 9, 0, 0, tzinfo=UTC))
    write_run(valid, tmp_path)
    (tmp_path / "20260525T120000-s.json").write_text("{garbage", encoding="utf-8")

    result = read_last_run("s", tmp_path)
    assert result is not None
    assert result.started_at == valid.started_at
