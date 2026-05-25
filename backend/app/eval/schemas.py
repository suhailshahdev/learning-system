"""Schemas for the evaluation framework.

An eval set is a versioned list of items scored against a target.
Three kinds exist, one per evaluator, and a set is homogeneous: every
item in a set shares one kind, because a set maps to exactly one
evaluator. So rather than a single item type with untyped input and
expected fields, each kind has its own item model and its own set
model. The loader dispatches on the set's eval_kind the way the
response parser dispatches on the first delimiter.

  parser     -> ParserEvalSet      scores parse_response() output
  retrieval  -> RetrievalEvalSet    scores search_corpus() ranking
  teaching   -> TeachingEvalSet     scores a teaching turn via a judge

Items are heterogeneous across kinds but scores are uniform: a score
is (item_id, outcome, detail) no matter which set produced it. That
seam keeps RunRecord single and non-generic while item inputs stay
fully typed.

The expected field for a parser item is itself a small discriminated
union: an input either parses to a known response kind, or raises
ParseError. Mirrors the kind-discriminator pattern the response union
already uses so mypy narrows in match blocks.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 (Pydantic runtime field resolution)
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# Pydantic v2 resolves field annotations at runtime, including the
# datetime, enums, and nested target union used as field types below.
# A TYPE_CHECKING-only import leaves them unresolvable when Pydantic
# builds the validator. Same constraint documented in parsed_response.py
# and the SQLAlchemy models. The codes differ by import category: TC003
# for the stdlib datetime, TC001 for the first-party app imports.
from app.eval.targets import TargetDescriptor  # noqa: TC001 (Pydantic runtime field resolution)
from app.models.enums import (  # noqa: TC001 (Pydantic runtime field resolution)
    Difficulty,
    LearningMode,
)


class EvalKind(StrEnum):
    """Which evaluator a set is scored by.

    The set's top-level discriminator. The loader reads this first to
    pick the concrete set model to validate against.
    """

    PARSER = "parser"
    RETRIEVAL = "retrieval"
    TEACHING = "teaching"


class ScoreOutcome(StrEnum):
    """The result of scoring one item.

    PASS and FAIL are the deterministic outcomes. ERROR means the
    item could not be scored at all: the target raised unexpectedly,
    the judge call failed, the fixture could not load. ERROR is kept
    distinct from FAIL so a broken run is not misread as a regression
    in the thing under test.
    """

    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"


# ---------------------------------------------------------------------------
# Parser eval set
# ---------------------------------------------------------------------------


class ParsesTo(BaseModel):
    """The input is expected to parse to a given response kind.

    kind matches the Literal discriminator on the ParsedResponse union
    members (turn, session_end, grading, handover, tool_call, proposal).
    fields is an optional subset of the parsed model's fields that must
    equal the parsed output. An empty fields dict asserts only that the
    input parsed to the right kind, nothing about its contents.
    """

    model_config = ConfigDict(frozen=True)

    outcome: Literal["parses_to"] = "parses_to"
    kind: Literal["turn", "session_end", "grading", "handover", "tool_call", "proposal"]
    fields: dict[str, str] = Field(default_factory=dict)


class RaisesParseError(BaseModel):
    """The input is expected to raise ParseError.

    message_contains, when set, asserts the raised error's message
    contains the substring. When None, any ParseError passes. Substring
    rather than exact match because parser messages embed variable
    detail (the offending value, the field index) that would make exact
    assertions brittle.
    """

    model_config = ConfigDict(frozen=True)

    outcome: Literal["raises"] = "raises"
    message_contains: str | None = None


type ParserExpectation = Annotated[
    ParsesTo | RaisesParseError,
    Field(discriminator="outcome"),
]


class ParserEvalItem(BaseModel):
    """One parser-robustness item: a raw wire string and its expectation.

    raw is fed directly to parse_response(). expected says whether that
    should succeed (and to what) or raise. provenance records where the
    item came from (hand-written, harvested from an error_log row) since
    that bears on what the item is evidence of.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    raw: str
    expected: ParserExpectation
    provenance: str | None = None
    tags: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Retrieval eval set
# ---------------------------------------------------------------------------


class RankAssertion(BaseModel):
    """One ranking expectation for a retrieval query.

    doc_id names a document in the set's fixture corpus. max_rank is the
    worst position (1-indexed) the doc may occupy in the results and
    still pass: max_rank=1 means it must rank first, max_rank=3 means
    top-three. min_score, when set, also requires the cosine similarity
    be at least that value, guarding against a doc ranking well only
    because the whole corpus scored poorly.
    """

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(min_length=1)
    max_rank: int = Field(ge=1)
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)


class RetrievalEvalItem(BaseModel):
    """One retrieval item: a query and the ranks its known docs must hit.

    The corpus is not on the item, it lives once on the set (the fixture
    every query in the set runs against). Each query asserts one or more
    RankAssertions over doc_ids in that corpus.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    expected: list[RankAssertion] = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)


class FixtureDoc(BaseModel):
    """One document in a retrieval set's fixture corpus.

    doc_id is referenced by RankAssertion.doc_id. content is embedded
    into the throwaway corpus at run time. Keeping the corpus in the set
    file makes expected ranks stable and version-controlled: the same
    fixture always embeds to the same relative distances, so a rank
    regression means the retrieval code changed, not that someone ran a
    study session.
    """

    model_config = ConfigDict(frozen=True)

    doc_id: str = Field(min_length=1)
    content: str = Field(min_length=1)


# ---------------------------------------------------------------------------
# Teaching-turn eval set
# ---------------------------------------------------------------------------


class TeachingSetup(BaseModel):
    """The session state a teaching item starts the LLM from.

    Enough to build the first-turn prompt: the topic to teach, the mode
    to teach in, and a free-text learner-state note the prompt layer
    folds into context. Deliberately thin. The judged output is the
    teaching turn the LLM produces from this setup, not a multi-turn
    transcript.
    """

    model_config = ConfigDict(frozen=True)

    topic_path: str = Field(min_length=1)
    mode: LearningMode
    difficulty: Difficulty
    learner_state: str | None = None


class TeachingEvalItem(BaseModel):
    """One teaching-turn item: a setup and the rubric to judge against.

    rubric is the natural-language scoring guide handed to the judge
    model. pass_threshold is the minimum mean score (over the run's N
    repeats) for the item to pass. N repeats and variance gating live on
    the run config, not the item, since they are a property of how the
    set is run, not of the item itself.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    setup: TeachingSetup
    rubric: str = Field(min_length=1)
    pass_threshold: float = Field(ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Eval sets
# ---------------------------------------------------------------------------


class _EvalSetBase(BaseModel):
    """Shared metadata for every eval set.

    name identifies the set across runs (the regression report pairs a
    run with the previous run of the same name). schema_version guards
    the set-file format itself, separate from the content hash: a format
    migration bumps schema_version, an item edit changes the content
    hash. The hash is not stored on the set, the loader computes it from
    the on-disk bytes so it cannot drift from the file it describes.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    schema_version: int = Field(ge=1)


class ParserEvalSet(_EvalSetBase):
    """A homogeneous set of parser-robustness items."""

    model_config = ConfigDict(frozen=True)

    eval_kind: Literal[EvalKind.PARSER] = EvalKind.PARSER
    items: list[ParserEvalItem] = Field(min_length=1)


class RetrievalEvalSet(_EvalSetBase):
    """A homogeneous set of retrieval items plus the fixture corpus they share."""

    model_config = ConfigDict(frozen=True)

    eval_kind: Literal[EvalKind.RETRIEVAL] = EvalKind.RETRIEVAL
    corpus: list[FixtureDoc] = Field(min_length=1)
    items: list[RetrievalEvalItem] = Field(min_length=1)


class TeachingEvalSet(_EvalSetBase):
    """A homogeneous set of teaching-turn items."""

    model_config = ConfigDict(frozen=True)

    eval_kind: Literal[EvalKind.TEACHING] = EvalKind.TEACHING
    items: list[TeachingEvalItem] = Field(min_length=1)


type EvalSet = Annotated[
    ParserEvalSet | RetrievalEvalSet | TeachingEvalSet,
    Field(discriminator="eval_kind"),
]


# ---------------------------------------------------------------------------
# Run record
# ---------------------------------------------------------------------------


class ItemScore(BaseModel):
    """The score for one item in one run.

    outcome is pass/fail/error. detail is human-readable: the failing
    field diff, the rank the doc actually hit, the judge's rationale.
    scores carries the raw per-repeat numbers for judged items (one
    entry per repeat) so the regression report can compute mean and
    variance. It is empty for deterministic items, which run once.
    """

    model_config = ConfigDict(frozen=True)

    item_id: str = Field(min_length=1)
    outcome: ScoreOutcome
    detail: str = ""
    scores: list[float] = Field(default_factory=list)


class RunRecord(BaseModel):
    """The record of one eval run, written to eval/runs/.

    set_name and set_content_hash pin which set and which version of it
    this run scored. The hash closes the versioning watchpoint: two runs
    of the same set name with different hashes scored different content,
    and the report can say so. target describes what the set ran against
    (the parser, the search service, or a transport+model). cost_usd is
    the run's total estimated spend, zero for free deterministic runs and
    non-zero once a judge or fixture embedding is involved.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(min_length=1)
    set_name: str = Field(min_length=1)
    set_content_hash: str = Field(min_length=1)
    eval_kind: EvalKind
    target: TargetDescriptor
    started_at: datetime
    finished_at: datetime
    scores: list[ItemScore]
    cost_usd: float = Field(default=0.0, ge=0.0)
