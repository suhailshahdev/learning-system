"""Tests for prereq_service."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from app.models import (
    AssertionSource,
    Difficulty,
    Topic,
    TopicStatus,
    UserKnowledgeAssertion,
    UserProfile,
)
from app.models.user_profile import SINGLETON_ID
from app.services.prereq_service import (
    PrereqsUnmetError,
    UnmetPrerequisite,
    check_prerequisites,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession


def _add_topic(
    db: DbSession,
    path: str,
    prerequisites: list[dict[str, str]] | None = None,
) -> Topic:
    topic = Topic(
        path=path,
        domain=path.split(" > ", 1)[0],
        name=path.rsplit(" > ", 1)[-1],
        status=TopicStatus.IN_PROGRESS,
        prerequisites=prerequisites or [],
    )
    db.add(topic)
    db.flush()
    return topic


def _add_assertion(
    db: DbSession,
    topic_path: str,
    difficulty: Difficulty,
    source: AssertionSource = AssertionSource.SELF_DECLARED,
) -> None:
    db.add(
        UserKnowledgeAssertion(
            topic_path=topic_path,
            difficulty=difficulty,
            source=source,
        )
    )
    db.flush()


def _set_dismissed(db: DbSession, paths: list[str]) -> None:
    profile = db.get(UserProfile, SINGLETON_ID)
    if profile is None:
        profile = UserProfile(id=SINGLETON_ID, prereq_warning_dismissed=paths)
        db.add(profile)
    else:
        profile.prereq_warning_dismissed = paths
    db.flush()


def test_topic_does_not_exist_returns_empty(db: DbSession) -> None:
    assert check_prerequisites(db, "Python > Async") == []


def test_topic_with_no_prerequisites_returns_empty(db: DbSession) -> None:
    _add_topic(db, "Python > Basics")
    assert check_prerequisites(db, "Python > Basics") == []


def test_unmet_prerequisite_with_no_assertion(db: DbSession) -> None:
    _add_topic(
        db,
        "FastAPI > Routing",
        prerequisites=[{"topic_path": "Python > Basics", "min_difficulty": "intermediate"}],
    )
    unmet = check_prerequisites(db, "FastAPI > Routing")
    assert unmet == [
        UnmetPrerequisite(
            topic_path="Python > Basics",
            min_difficulty=Difficulty.INTERMEDIATE,
            asserted_difficulty=None,
        )
    ]


def test_unmet_prerequisite_with_lower_assertion(db: DbSession) -> None:
    _add_topic(
        db,
        "FastAPI > Routing",
        prerequisites=[{"topic_path": "Python > Basics", "min_difficulty": "intermediate"}],
    )
    _add_assertion(db, "Python > Basics", Difficulty.BEGINNER)
    unmet = check_prerequisites(db, "FastAPI > Routing")
    assert unmet == [
        UnmetPrerequisite(
            topic_path="Python > Basics",
            min_difficulty=Difficulty.INTERMEDIATE,
            asserted_difficulty=Difficulty.BEGINNER,
        )
    ]


def test_met_prerequisite_at_exact_difficulty(db: DbSession) -> None:
    _add_topic(
        db,
        "FastAPI > Routing",
        prerequisites=[{"topic_path": "Python > Basics", "min_difficulty": "intermediate"}],
    )
    _add_assertion(db, "Python > Basics", Difficulty.INTERMEDIATE)
    assert check_prerequisites(db, "FastAPI > Routing") == []


def test_met_prerequisite_at_higher_difficulty(db: DbSession) -> None:
    _add_topic(
        db,
        "FastAPI > Routing",
        prerequisites=[{"topic_path": "Python > Basics", "min_difficulty": "intermediate"}],
    )
    _add_assertion(db, "Python > Basics", Difficulty.ADVANCED)
    assert check_prerequisites(db, "FastAPI > Routing") == []


def test_multiple_assertions_take_highest_difficulty(db: DbSession) -> None:
    _add_topic(
        db,
        "FastAPI > Routing",
        prerequisites=[{"topic_path": "Python > Basics", "min_difficulty": "intermediate"}],
    )
    _add_assertion(db, "Python > Basics", Difficulty.BEGINNER, AssertionSource.SELF_DECLARED)
    _add_assertion(
        db,
        "Python > Basics",
        Difficulty.ADVANCED,
        AssertionSource.DERIVED_FROM_LEARNED_ITEMS,
    )
    assert check_prerequisites(db, "FastAPI > Routing") == []


def test_dismissed_warning_returns_empty(db: DbSession) -> None:
    _add_topic(
        db,
        "FastAPI > Routing",
        prerequisites=[{"topic_path": "Python > Basics", "min_difficulty": "intermediate"}],
    )
    _set_dismissed(db, ["FastAPI > Routing"])
    assert check_prerequisites(db, "FastAPI > Routing") == []


def test_multiple_prereqs_some_met_some_not(db: DbSession) -> None:
    _add_topic(
        db,
        "FastAPI > Background Tasks",
        prerequisites=[
            {"topic_path": "Python > Async", "min_difficulty": "intermediate"},
            {"topic_path": "FastAPI > Routing", "min_difficulty": "beginner"},
        ],
    )
    _add_assertion(db, "FastAPI > Routing", Difficulty.BEGINNER)
    unmet = check_prerequisites(db, "FastAPI > Background Tasks")
    assert unmet == [
        UnmetPrerequisite(
            topic_path="Python > Async",
            min_difficulty=Difficulty.INTERMEDIATE,
            asserted_difficulty=None,
        )
    ]


def test_prereqs_unmet_error_carries_the_list() -> None:
    unmet = [
        UnmetPrerequisite(
            topic_path="Python > Basics",
            min_difficulty=Difficulty.INTERMEDIATE,
            asserted_difficulty=None,
        )
    ]
    with pytest.raises(PrereqsUnmetError) as exc_info:
        raise PrereqsUnmetError(unmet)
    assert exc_info.value.unmet == unmet
