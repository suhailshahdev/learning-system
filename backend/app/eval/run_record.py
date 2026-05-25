"""Write eval run records and find the previous run for a set.

A run record is a JSON file under eval/runs/. write_run serializes one
record. read_last_run finds the most recent record for a given set name
so the regression report can diff this run against it. There is no DB
table: run records are append-only artifacts, never queried relationally,
and they must survive a db reset (the corpus-cleanup gate runs one right
before the retrieval set). Files travel with the repo and diff in git.

"Most recent" is resolved from each record's started_at field, not from
filesystem mtime (a copied file lies) or filename sort alone (same-second
runs collide). The filename carries a timestamp prefix for human
readability and to narrow the glob. The record's own field is the truth.

Set-name matching is exact on the record's set_name field, not a filename
substring: a set named "parser" must not match "parser_robustness".
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from app.eval.schemas import RunRecord

if TYPE_CHECKING:
    from pathlib import Path

# Characters allowed in the set-name portion of a filename. Anything else
# in a set name is replaced so the filename stays portable. The collapse
# is one-directional and lossy by design: the record's set_name field is
# authoritative, the filename is a convenience, so a slugged filename
# never needs to round-trip back to the exact name.
_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9_-]+")

# Basic-ISO timestamp for the filename prefix: sortable, no colons (which
# are illegal in filenames on some systems). Distinct from the record's
# stored started_at, which keeps full ISO format inside the JSON.
_TS_FORMAT = "%Y%m%dT%H%M%S"


def write_run(record: RunRecord, runs_dir: Path) -> Path:
    """Write a run record to runs_dir, returning the path written.

    Creates runs_dir if it does not exist. The filename is the run's
    start timestamp followed by a filename-safe form of the set name, so
    a directory listing is chronological and greppable by set.
    """
    runs_dir.mkdir(parents=True, exist_ok=True)
    ts = record.started_at.strftime(_TS_FORMAT)
    safe_name = _FILENAME_SAFE_RE.sub("-", record.set_name)
    path = runs_dir / f"{ts}-{safe_name}.json"
    path.write_text(record.model_dump_json(indent=2), encoding="utf-8")
    return path


def read_last_run(set_name: str, runs_dir: Path) -> RunRecord | None:
    """Return the most recent run record for a set, or None if none exist.

    Reads every record in runs_dir, keeps those whose set_name field
    exactly matches, and returns the one with the latest started_at.
    Returns None when runs_dir is absent or holds no matching record.

    Malformed files in runs_dir are skipped, not fatal: a hand-edited or
    truncated record should not block reading a valid previous run. The
    skip is silent because runs_dir is a write-only-by-us artifact dir.
    A malformed file there is not a user-facing error the way a malformed
    eval set is.
    """
    if not runs_dir.is_dir():
        return None

    latest: RunRecord | None = None
    for path in runs_dir.glob("*.json"):
        record = _try_read(path)
        if record is None or record.set_name != set_name:
            continue
        if latest is None or record.started_at > latest.started_at:
            latest = record
    return latest


def _try_read(path: Path) -> RunRecord | None:
    """Read one run record, returning None if it cannot be parsed.

    Used by read_last_run to skip malformed files rather than fail the
    whole lookup. A run record we wrote validates, a file that does not
    is something else and is ignored.
    """
    try:
        text = path.read_text(encoding="utf-8")
        return RunRecord.model_validate_json(text)
    except (OSError, ValueError):
        return None
