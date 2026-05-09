"""Shared schema fragments used across multiple schema modules.

This module exists to break circular imports. Types defined here
are wire-format primitives that appear in more than one schema:
the parser uses them in ParsedTurn, the tool surface uses them in
CreateOrUpdateTopicInput, and so on. Putting them in a leaf module
that depends on nothing else keeps the dependency graph acyclic.
"""

from __future__ import annotations

# Pydantic v2 fails to resolve generic types with TYPE_CHECKING-only
# imports, runtime imports are required for any field type Pydantic
# evaluates at model construction. Same constraint as parsed_response.py
# and tools.py.
from app.models.enums import Difficulty  # noqa: TC002
from pydantic import BaseModel, ConfigDict, Field


class Prerequisite(BaseModel):
    """One prerequisite entry, used in topic metadata and parsed turns.

    The wire format is `topic_path:difficulty` per pair, comma-
    separated. The parser splits and validates so consumers see
    structured data and can run prereq checks without re-parsing
    strings.
    """

    model_config = ConfigDict(frozen=True)

    topic_path: str = Field(min_length=1)
    min_difficulty: Difficulty
