"""Prerequisite check.

Reads a topic's prerequisites and the user's knowledge assertions,
and returns any prerequisites the user has not yet satisfied. The
session engine calls this at session start and the route layer
returns a 409 with the unmet list so the UI can show the prereq
modal.

Prerequisites come from topic.prerequisites, which is populated
after the first session on a topic. On the very first session the
check returns nothing since no data exists yet.

A higher difficulty satisfies a lower one, so an intermediate
assertion covers a beginner prerequisite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from app.models import Difficulty, Topic, UserKnowledgeAssertion, UserProfile
from app.models.user_profile import SINGLETON_ID
from app.schemas.parsed_response import Prerequisite

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession


class UnmetPrerequisite(BaseModel):
    """One prereq the user has not satisfied for the target topic."""

    model_config = ConfigDict(frozen=True)

    topic_path: str
    min_difficulty: Difficulty
    asserted_difficulty: Difficulty | None


class PrereqsUnmetError(Exception):
    """Raised when starting a session would require unmet prerequisites.

    Carries the list so the route layer can surface it in the 409
    response. The session service raises this before any transport
    call, so rollback is a no-op.
    """

    def __init__(self, unmet: list[UnmetPrerequisite]) -> None:
        super().__init__(f"{len(unmet)} unmet prerequisite(s).")
        self.unmet = unmet


_DIFFICULTY_ORDER: dict[Difficulty, int] = {
    Difficulty.BEGINNER: 0,
    Difficulty.INTERMEDIATE: 1,
    Difficulty.ADVANCED: 2,
}


def _difficulty_satisfies(actual: Difficulty, required: Difficulty) -> bool:
    """True if `actual` is at least `required`."""
    return _DIFFICULTY_ORDER[actual] >= _DIFFICULTY_ORDER[required]


def check_prerequisites(db: DbSession, topic_path: str) -> list[UnmetPrerequisite]:
    """Return prerequisites the user has not yet satisfied for this topic.

    Returns an empty list when:
    - the topic does not exist yet (first session on it)
    - the topic exists but declares no prerequisites
    - the user has dismissed the prereq warning for this topic
    - all prerequisites are satisfied
    """
    profile = db.get(UserProfile, SINGLETON_ID)
    dismissed = set(profile.prereq_warning_dismissed) if profile is not None else set()
    if topic_path in dismissed:
        return []

    topic = db.query(Topic).filter(Topic.path == topic_path).one_or_none()
    if topic is None or not topic.prerequisites:
        return []

    asserted_at = _highest_asserted_difficulty_by_path(db)

    unmet: list[UnmetPrerequisite] = []
    for raw in topic.prerequisites:
        prereq = Prerequisite.model_validate(raw)
        actual = asserted_at.get(prereq.topic_path)
        if actual is None or not _difficulty_satisfies(actual, prereq.min_difficulty):
            unmet.append(
                UnmetPrerequisite(
                    topic_path=prereq.topic_path,
                    min_difficulty=prereq.min_difficulty,
                    asserted_difficulty=actual,
                )
            )
    return unmet


def _highest_asserted_difficulty_by_path(db: DbSession) -> dict[str, Difficulty]:
    """Build a {topic_path: highest asserted difficulty} map.

    Multiple assertions can exist per path (e.g. one self-declared
    at beginner, one derived at intermediate). The highest wins for
    prereq satisfaction.
    """
    assertions = db.query(UserKnowledgeAssertion).all()
    highest: dict[str, Difficulty] = {}
    for a in assertions:
        current = highest.get(a.topic_path)
        if current is None or _DIFFICULTY_ORDER[a.difficulty] > _DIFFICULTY_ORDER[current]:
            highest[a.topic_path] = a.difficulty
    return highest
