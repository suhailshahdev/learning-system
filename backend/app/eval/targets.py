"""What an eval run executed against.

A run record pins not just which set ran but what it ran against, so a
score is interpretable later: a parser run has no model, a retrieval run
embeds against one embedding model, a teaching run drives one transport
with one chat model and is judged by a separate one. TargetDescriptor is
the typed record of that, stored on RunRecord and printed in the report.

The separate-judge constraint lives here as a model validator rather than
in the runner: the judge model must differ from the model under test, or
scores skew up (a model grading its own output rates it generously). A
config that pairs them is a mistake to catch at load time, before any
money is spent, not a runtime surprise.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TargetKind(StrEnum):
    """The class of thing a set ran against.

    PARSER and SEARCH are in-process pure-ish targets with no LLM. JUDGE
    is the only kind that drives a transport and a judge model, so it is
    the only kind carrying model fields.
    """

    PARSER = "parser"
    SEARCH = "search"
    JUDGE = "judge"


class ParserTarget(BaseModel):
    """The parser run against parse_response(). No model, no cost."""

    model_config = ConfigDict(frozen=True)

    kind: Literal[TargetKind.PARSER] = TargetKind.PARSER


class SearchTarget(BaseModel):
    """The search service run against a fixture corpus.

    embedding_model names the embedder the fixture was embedded with and
    the query was embedded against. Recorded because a retrieval score is
    only comparable across runs that used the same embedding model: swap
    the embedder and the distances move, which is a legitimate thing to
    detect but must not be misread as a retrieval-code regression.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal[TargetKind.SEARCH] = TargetKind.SEARCH
    embedding_model: str = Field(min_length=1)


class JudgeTarget(BaseModel):
    """A teaching run: one transport+model under test, one judge model.

    transport names which transport drove the teaching turn (the kind
    string, deepseek or playwright). model_under_test is the chat model
    that produced the turn. judge_model scores it. The validator enforces
    that judge_model differs from model_under_test so a model never grades
    its own output.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal[TargetKind.JUDGE] = TargetKind.JUDGE
    transport: str = Field(min_length=1)
    model_under_test: str = Field(min_length=1)
    judge_model: str = Field(min_length=1)

    @model_validator(mode="after")
    def _judge_differs_from_subject(self) -> Self:
        """Reject a judge that is the same model as the one under test."""
        if self.judge_model == self.model_under_test:
            msg = (
                f"judge_model must differ from model_under_test; both are "
                f"{self.model_under_test!r}. A model grading its own output "
                f"skews scores up."
            )
            raise ValueError(msg)
        return self


type TargetDescriptor = Annotated[
    ParserTarget | SearchTarget | JudgeTarget,
    Field(discriminator="kind"),
]
