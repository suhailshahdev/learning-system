"""Deterministic evaluator for parser-robustness items.

Runs parse_response on an item's raw wire string and scores the outcome
against the item's expectation: either the input parses to a named
response kind (optionally with field assertions), or it raises ParseError
(optionally with a substring the message must contain).

Field assertions compare scalar values only. The parsed model is dumped
to JSON-mode primitives and each expected field is compared as a string:
this covers topic_path, difficulty, mode, question, and the other scalar
fields a parser item realistically pins. Structural fields (prerequisites,
code blocks) are not meant to be asserted through a set's fields dict,
they are exercised in the evaluator's own tests against the model. A set
that needs structural assertions is a signal to add a typed expectation
shape, not to deep-compare JSON here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.eval.schemas import (
    ItemScore,
    ParserEvalItem,
    ParserExpectation,
    ScoreOutcome,
)
from app.services.parser import ParseError, parse_response

if TYPE_CHECKING:
    from app.schemas.parsed_response import ParsedResponse


def evaluate_parser_item(item: ParserEvalItem) -> ItemScore:
    """Score one parser item by running the parser and matching the outcome.

    Never raises on a parse failure: a ParseError from parse_response is
    data to be matched against the expectation, not an error in the
    evaluator. An ERROR outcome is reserved for the evaluator itself
    failing unexpectedly, which here means an expectation we cannot
    interpret.
    """
    expected = item.expected

    try:
        parsed = parse_response(item.raw)
    except ParseError as e:
        return _score_against_raise(item.id, expected, raised=e)

    return _score_against_parse(item.id, expected, parsed=parsed)


def _score_against_parse(
    item_id: str, expected: ParserExpectation, parsed: ParsedResponse
) -> ItemScore:
    """Score when the parser returned a result (did not raise)."""
    if expected.outcome == "raises":
        return ItemScore(
            item_id=item_id,
            outcome=ScoreOutcome.FAIL,
            detail=f"expected ParseError, but parsed to kind {parsed.kind!r}",
        )

    # expected.outcome == "parses_to"
    if parsed.kind != expected.kind:
        return ItemScore(
            item_id=item_id,
            outcome=ScoreOutcome.FAIL,
            detail=f"expected kind {expected.kind!r}, parsed to {parsed.kind!r}",
        )

    mismatch = _first_field_mismatch(parsed, expected.fields)
    if mismatch is not None:
        return ItemScore(item_id=item_id, outcome=ScoreOutcome.FAIL, detail=mismatch)

    return ItemScore(item_id=item_id, outcome=ScoreOutcome.PASS)


def _score_against_raise(
    item_id: str, expected: ParserExpectation, raised: ParseError
) -> ItemScore:
    """Score when the parser raised ParseError."""
    if expected.outcome == "parses_to":
        return ItemScore(
            item_id=item_id,
            outcome=ScoreOutcome.FAIL,
            detail=f"expected parse to {expected.kind!r}, but raised: {raised.message}",
        )

    # expected.outcome == "raises"
    if expected.message_contains is not None and expected.message_contains not in raised.message:
        return ItemScore(
            item_id=item_id,
            outcome=ScoreOutcome.FAIL,
            detail=(
                f"raised, but message {raised.message!r} "
                f"does not contain {expected.message_contains!r}"
            ),
        )

    return ItemScore(item_id=item_id, outcome=ScoreOutcome.PASS)


def _first_field_mismatch(parsed: ParsedResponse, fields: dict[str, str]) -> str | None:
    """Return a description of the first field that does not match, or None.

    Compares each expected field against the parsed model dumped to
    JSON-mode primitives, as strings. Returns None when every expected
    field matches (or when fields is empty, asserting only the kind).
    """
    if not fields:
        return None

    dumped = parsed.model_dump(mode="json")
    for key, expected_value in fields.items():
        if key not in dumped:
            return f"expected field {key!r} not present in parsed {parsed.kind!r}"
        actual = str(dumped[key])
        if actual != expected_value:
            return f"field {key!r}: expected {expected_value!r}, got {actual!r}"
    return None
