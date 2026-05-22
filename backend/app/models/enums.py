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
    """Who produced a turn. Used by `session_turn.role`.

    TRANSITION marks the boundary between the old chat and the new
    chat when a session crosses the per-chat message threshold. The
    turn's raw_content holds the handover block that was carried
    over.

    TOOL_CALL and TOOL_RESULT come in pairs around the
    session-service tool-execution loop. TOOL_CALL records the LLM
    requesting a tool with the raw call envelope in raw_content and
    the validated ToolCall in parsed. TOOL_RESULT records the
    handler's output sent back to the LLM. Both count toward the
    per-chat message budget since they consume real turn space in
    every transport.

    GRADING records the LLM's standalone grading response.
    A teaching cycle now persists as USER (answer) -> GRADING
    (LLM's grading) -> USER (continue prompt) -> ASSISTANT
    (next teaching turn). approve_session's _build_learned_items
    skips GRADING turns when pairing teaching turns with user
    answers, same pattern it already uses for TRANSITION.
    """

    ASSISTANT = "assistant"
    USER = "user"
    SYSTEM = "system"
    TRANSITION = "transition"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    GRADING = "grading"


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


class GradingVerdict(StrEnum):
    """Verdict on a user's previous answer. Used by `parsed_response.ParsedTurn.grading_verdict`.

    The LLM emits a verdict on each turn that has a previous answer
    to grade. The first turn of a session has no previous answer,
    so the field is None on first turns. open_graded covers modes
    where the answer is free-form prose (explain_back, socratic);
    the verdict is informational and the explanation carries the
    teaching feedback.
    """

    CORRECT = "correct"
    PARTIAL = "partial"
    INCORRECT = "incorrect"
    OPEN_GRADED = "open_graded"


class TransportKind(StrEnum):
    """Which LLM transport opened a chat. Used by `session.transport_kind`.

    A follow-up turn needs to know which transport's resume_chat to
    call; the column makes that routing decision durable.
    """

    CLAUDE_PLAYWRIGHT = "claude_playwright"
    DEEPSEEK = "deepseek"


class EmbeddingSourceType(StrEnum):
    """What an embedding row was derived from. Used by `embedding.source_type`.

    Two sources at present. LEARNED_ITEM embeddings come from an
    approved question/answer pair, source_id is a learned_item.id.
    DOCUMENT_CHUNK embeddings come from a slice of an ingested text
    document, source_id is a document.id. source_id is a polymorphic
    reference resolved by source_type, not a foreign key, since it
    points at different tables per kind.
    """

    LEARNED_ITEM = "learned_item"
    DOCUMENT_CHUNK = "document_chunk"
