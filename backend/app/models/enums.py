"""Shared enums used across models.

Each enum maps to a database
column or is used by more than one model, so they all live in
one file for easy reference.

All enums inherit from StrEnum, meaning each value is also a plain
string. This keeps the database, API, and frontend in sync without
any extra conversion.
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


class SessionState(StrEnum):
    """Lifecycle state of a session. Used by `session.state`."""

    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ABANDONED = "abandoned"
    ARCHIVED = "archived"


class TurnRole(StrEnum):
    """Who produced a turn. Used by `session_turn.role`."""

    CLAUDE = "claude"
    USER = "user"
    SYSTEM = "system"


class LearnedItemStatus(StrEnum):
    """Status of a learned item. Used by `learned_item.status`."""

    LEARNED = "learned"
    NEEDS_REVISION = "needs_revision"


class LearningMode(StrEnum):
    """Learning mode for a question. Used by `session.mode_used`
    and `session_turn.mode
    """

    FLASHCARD = "flashcard"
    TYPE_THE_ANSWER = "type_the_answer"
    CODE_WITH_EXPLANATION = "code_with_explanation"
    MULTIPLE_CHOICE = "multiple_choice"
    EXPLAIN_BACK = "explain_back"
    SOCRATIC = "socratic"
