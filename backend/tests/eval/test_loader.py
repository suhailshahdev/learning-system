"""Tests for the eval-set loader and content hashing.

Loading covers the happy path (valid set round-trips to the right
concrete type) and each failure mode (missing file, bad JSON, schema
mismatch) raising EvalSetLoadError. Hashing covers the four properties
that make the hash useful: stable across file reformatting, stable
across a metadata change, and moving when an item or the corpus changes.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from app.eval.loader import EvalSetLoadError, content_hash, load_set
from app.eval.schemas import ParserEvalSet, RetrievalEvalSet

if TYPE_CHECKING:
    from pathlib import Path


def _parser_set_dict() -> dict[str, object]:
    """A minimal valid parser set as a JSON-ready dict."""
    return {
        "eval_kind": "parser",
        "name": "parser_smoke",
        "schema_version": 1,
        "items": [
            {
                "id": "p1",
                "raw": "---TOPIC---\nPython > Basics\n---END---",
                "expected": {"outcome": "raises", "message_contains": "fields"},
            },
            {
                "id": "p2",
                "raw": "---SESSION_END_PROPOSAL---\nAll done.\n---END---",
                "expected": {"outcome": "parses_to", "kind": "session_end"},
            },
        ],
    }


def _retrieval_set_dict() -> dict[str, object]:
    """A minimal valid retrieval set with a two-doc fixture corpus."""
    return {
        "eval_kind": "retrieval",
        "name": "retrieval_smoke",
        "schema_version": 1,
        "corpus": [
            {"doc_id": "d1", "content": "How to append to a list in Python."},
            {"doc_id": "d2", "content": "How to define a class in Python."},
        ],
        "items": [
            {
                "id": "r1",
                "query": "adding an element to a list",
                "expected": [{"doc_id": "d1", "max_rank": 1}],
            },
        ],
    }


def _write(tmp_path: Path, data: dict[str, object]) -> Path:
    """Write a dict as JSON to a temp file and return the path."""
    path = tmp_path / "set.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_parser_set_returns_concrete_type(tmp_path: Path) -> None:
    path = _write(tmp_path, _parser_set_dict())
    result = load_set(path)
    assert isinstance(result, ParserEvalSet)
    assert result.name == "parser_smoke"
    assert len(result.items) == 2


def test_load_retrieval_set_returns_concrete_type(tmp_path: Path) -> None:
    path = _write(tmp_path, _retrieval_set_dict())
    result = load_set(path)
    assert isinstance(result, RetrievalEvalSet)
    assert len(result.corpus) == 2


def test_load_missing_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "does_not_exist.json"
    with pytest.raises(EvalSetLoadError) as exc:
        load_set(path)
    assert "could not read file" in exc.value.message


def test_load_bad_json_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(EvalSetLoadError) as exc:
        load_set(path)
    assert "not valid JSON" in exc.value.message


def test_load_schema_mismatch_raises(tmp_path: Path) -> None:
    data = _parser_set_dict()
    del data["items"]
    path = _write(tmp_path, data)
    with pytest.raises(EvalSetLoadError) as exc:
        load_set(path)
    assert "failed schema validation" in exc.value.message


def test_load_unknown_eval_kind_raises(tmp_path: Path) -> None:
    data = _parser_set_dict()
    data["eval_kind"] = "nonsense"
    path = _write(tmp_path, data)
    with pytest.raises(EvalSetLoadError):
        load_set(path)


def test_hash_stable_across_reformatting(tmp_path: Path) -> None:
    data = _parser_set_dict()
    compact = _write(tmp_path, data)
    pretty_path = tmp_path / "pretty.json"
    pretty_path.write_text(json.dumps(data, indent=4, sort_keys=False), encoding="utf-8")

    assert content_hash(load_set(compact)) == content_hash(load_set(pretty_path))


def test_hash_stable_across_name_change(tmp_path: Path) -> None:
    base = load_set(_write(tmp_path, _parser_set_dict()))

    renamed_data = _parser_set_dict()
    renamed_data["name"] = "a_different_name"
    renamed_data["schema_version"] = 2
    renamed = load_set(_write(tmp_path, renamed_data))

    assert content_hash(base) == content_hash(renamed)


def test_hash_moves_on_item_change(tmp_path: Path) -> None:
    base = load_set(_write(tmp_path, _parser_set_dict()))

    edited_data = _parser_set_dict()
    items = edited_data["items"]
    assert isinstance(items, list)
    items[0]["id"] = "p1_renamed"
    edited = load_set(_write(tmp_path, edited_data))

    assert content_hash(base) != content_hash(edited)


def test_hash_moves_on_corpus_change(tmp_path: Path) -> None:
    base = load_set(_write(tmp_path, _retrieval_set_dict()))

    edited_data = _retrieval_set_dict()
    corpus = edited_data["corpus"]
    assert isinstance(corpus, list)
    corpus[0]["content"] = "A completely different sentence."
    edited = load_set(_write(tmp_path, edited_data))

    assert content_hash(base) != content_hash(edited)
