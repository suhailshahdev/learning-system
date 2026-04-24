"""Shared enums used across models.

This file holds every enum that either maps to a database column
or gets used by more than one model. Keeping them together means
you can open one file to see every valid value for every enum
column in the schema.

All enums inherit from `StrEnum`, so each member is both an enum
and a plain string. That matters because the database stores the
string value (for example, `"language"`) and the JSON API sends
the same string back to the frontend. No extra conversion step,
no surprise object wrappers.
"""

from __future__ import annotations

from enum import StrEnum


class DomainKind(StrEnum):
    """Kind of top-level domain. Used by `domain.kind`."""

    LANGUAGE = "language"
    FRAMEWORK = "framework"
    LIBRARY = "library"
    CONCEPT = "concept"
    TOOL = "tool"
    PRACTICE = "practice"
    OTHER = "other"


class Difficulty(StrEnum):
    """Difficulty level for topics, learned items, and knowledge assertions."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class AssertionSource(StrEnum):
    """Where a knowledge assertion came from. Used by `user_knowledge_assertion.source`."""

    SELF_DECLARED = "self_declared"
    DERIVED_FROM_LEARNED_ITEMS = "derived_from_learned_items"
    RESUME = "resume"
    JD = "jd"


class TopicStatus(StrEnum):
    """Session-level status of a topic. Used by `topic.status`

    Tracks whether a topic has been covered in a session and whether
    it's complete or needs another pass. Distinct from knowledge
    assertions, which track what the user knows on a broader level.
    """

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    LEARNED = "learned"
    NEEDS_REVISION = "needs_revision"
