"""Load and content-hash eval sets.

An eval set on disk is JSON. load_set reads it, validates it against the
discriminated EvalSet union, and returns the concrete set type (parser,
retrieval, or teaching) with full typing intact. Validation dispatches on
the eval_kind field through a TypeAdapter, the same pattern the response
parser uses for the ToolCall union.

content_hash answers the versioning watchpoint: which version of a set
did a run score. It hashes the set's content (items, and the corpus for
retrieval sets) and excludes metadata (name, schema_version, eval_kind).
Renaming a set or bumping its schema version leaves the hash stable. Only
an edit to an item or a fixture doc moves it. The regression report pairs
runs by set name separately, so the hash's only job is detecting content
change.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from pydantic import TypeAdapter, ValidationError

from app.eval.schemas import EvalSet

if TYPE_CHECKING:
    from pathlib import Path

# Built once: TypeAdapter compilation is the expensive part, so the same
# instance validates every set. Mirrors parser.py's _TOOL_CALL_ADAPTER.
_EVAL_SET_ADAPTER: TypeAdapter[EvalSet] = TypeAdapter(EvalSet)

# Keys excluded from the content hash. Metadata, not scored content.
_HASH_EXCLUDED_KEYS = ("name", "schema_version", "eval_kind")


class EvalSetLoadError(Exception):
    """An eval set file could not be read or validated.

    Carries the path so the CLI can report which file failed, and the
    underlying cause (a JSON error or a Pydantic ValidationError) for
    detail. A loud failure here is correct: a malformed eval set must
    not run as if empty or partial.
    """

    def __init__(self, path: Path, message: str, cause: Exception | None = None) -> None:
        super().__init__(f"{path}: {message}")
        self.path = path
        self.message = message
        self.cause = cause


def load_set(path: Path) -> EvalSet:
    """Read and validate an eval set from a JSON file.

    Dispatches on the eval_kind discriminator to the concrete set type.
    Raises EvalSetLoadError if the file is missing, not valid JSON, or
    does not match the schema.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise EvalSetLoadError(path, "could not read file", cause=e) from e

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise EvalSetLoadError(path, f"not valid JSON: {e.msg}", cause=e) from e

    try:
        return _EVAL_SET_ADAPTER.validate_python(data)
    except ValidationError as e:
        raise EvalSetLoadError(path, "failed schema validation", cause=e) from e


def content_hash(eval_set: EvalSet) -> str:
    """Return a sha256 over a set's scored content, excluding metadata.

    Stable against file formatting (key order, whitespace) because it
    hashes the canonical JSON of the model dump, not the file bytes.
    Stable against metadata changes (name, schema_version) because those
    keys are dropped. Moves when any item or fixture doc changes.
    """
    dumped = eval_set.model_dump(mode="json")
    payload = {k: v for k, v in dumped.items() if k not in _HASH_EXCLUDED_KEYS}
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
