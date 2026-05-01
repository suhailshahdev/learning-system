"""Knowledge derivation.

Walks a just-approved session's learned items and writes
derived_from_learned_items assertions when the user has enough
items at a given topic and difficulty to claim knowledge at that
level. The "What I know" view reads these alongside self-declared,
resume, and JD assertions.

Derivation runs inside approve_session's transaction so the items
and the assertions they produce always commit or roll back
together.

The count is cumulative across sessions, so three sessions of one
item each can trigger a derivation on the third approval.

The asserted difficulty is the highest level where the count meets
the threshold. The prereq service's higher-satisfies-lower logic
then handles the rest automatically.

Only derived_from_learned_items rows are touched. Self-declared,
resume, and JD assertions are never modified by this service.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import func

from app.models import (
    AssertionSource,
    Difficulty,
    LearnedItem,
    Topic,
    UserKnowledgeAssertion,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession

    from app.models.session import Session


# Minimum learned items at a given topic and difficulty before the
# system asserts knowledge at that level. Tuned empirically and can
# be moved to settings if a different value is needed elsewhere.
DERIVATION_THRESHOLD = 3


# Highest-first ordering for picking the asserted difficulty when a
# topic has met threshold at multiple levels.
_DIFFICULTY_DESCENDING: tuple[Difficulty, ...] = (
    Difficulty.ADVANCED,
    Difficulty.INTERMEDIATE,
    Difficulty.BEGINNER,
)


def derive_assertions_for_session(db: DbSession, session: Session) -> list[UserKnowledgeAssertion]:
    """Upsert derived knowledge assertions for the topics this session touched.

    Returns the assertions created or updated. An empty list means
    nothing crossed the threshold or every relevant topic was already
    at the right derived level. The caller is responsible for
    committing the transaction.
    """
    # Query LearnedItem directly rather than walking session.learned_items.
    # The relationship loader caches state and an unloaded collection does
    # not auto-populate from in-progress db.add() calls, even after flush.
    # Same reasoning as _next_turn_index in session_service.
    touched_topic_ids = {
        topic_id
        for (topic_id,) in db.query(LearnedItem.topic_id)
        .filter(LearnedItem.session_id == session.id)
        .distinct()
        .all()
    }
    if not touched_topic_ids:
        return []

    upserts: list[UserKnowledgeAssertion] = []
    for topic_id in touched_topic_ids:
        topic = db.get(Topic, topic_id)
        if topic is None:
            continue
        target_difficulty = _highest_difficulty_meeting_threshold(db, topic_id)
        if target_difficulty is None:
            continue
        upserted = _upsert_derived_assertion(db, topic.path, target_difficulty)
        if upserted is not None:
            upserts.append(upserted)

    # Flush so subsequent queries see the upserts. The conftest configures
    # autoflush=False, which means SELECTs do not auto-flush pending adds.
    # Service convention across the codebase: if you add rows, flush before
    # returning. _add_item, _get_or_create_topic both follow this pattern.
    if upserts:
        db.flush()
    return upserts


def _highest_difficulty_meeting_threshold(db: DbSession, topic_id: str) -> Difficulty | None:
    """Return the highest difficulty where the topic has >= threshold items.

    Null-difficulty items don't count. Returns None if no level
    meets threshold.
    """
    rows = (
        db.query(LearnedItem.difficulty, func.count(LearnedItem.id))
        .filter(LearnedItem.topic_id == topic_id)
        .filter(LearnedItem.difficulty.is_not(None))
        .group_by(LearnedItem.difficulty)
        .all()
    )
    # The is_not(None) filter guarantees difficulty is non-null at runtime,
    # but mypy cannot narrow through SQLAlchemy filter chains. Build the
    # dict with explicit non-null filtering so the type checker agrees.
    counts: dict[Difficulty, int] = {d: c for d, c in rows if d is not None}

    for difficulty in _DIFFICULTY_DESCENDING:
        if counts.get(difficulty, 0) >= DERIVATION_THRESHOLD:
            return difficulty
    return None


def _upsert_derived_assertion(
    db: DbSession, topic_path: str, difficulty: Difficulty
) -> UserKnowledgeAssertion | None:
    """Create or update the derived assertion for this topic_path.

    Only the derived_from_learned_items row is touched. Returns the
    assertion if it was created or upgraded, or None if an existing
    derived assertion already holds an equal or higher difficulty.
    """
    existing = (
        db.query(UserKnowledgeAssertion)
        .filter(UserKnowledgeAssertion.topic_path == topic_path)
        .filter(UserKnowledgeAssertion.source == AssertionSource.DERIVED_FROM_LEARNED_ITEMS)
        .one_or_none()
    )

    if existing is None:
        new_assertion = UserKnowledgeAssertion(
            topic_path=topic_path,
            difficulty=difficulty,
            source=AssertionSource.DERIVED_FROM_LEARNED_ITEMS,
        )
        db.add(new_assertion)
        return new_assertion

    if _difficulty_rank(difficulty) > _difficulty_rank(existing.difficulty):
        existing.difficulty = difficulty
        return existing
    return None


_DIFFICULTY_RANK: dict[Difficulty, int] = {
    Difficulty.BEGINNER: 0,
    Difficulty.INTERMEDIATE: 1,
    Difficulty.ADVANCED: 2,
}


def _difficulty_rank(d: Difficulty) -> int:
    """Numeric rank for comparison."""
    return _DIFFICULTY_RANK[d]
