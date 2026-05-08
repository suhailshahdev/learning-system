"""Tests for the topics view service.

Each test seeds rows directly to control field values precisely.
Factories are duplicated from test_home_service.py rather than
imported because reaching across test files for helpers couples
the two suites.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.models import (
    Difficulty,
    Domain,
    DomainKind,
    Topic,
    TopicStatus,
)
from app.services.topics_service import build_topics_response

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession


def _make_topic(
    db: DbSession,
    *,
    path: str,
    status: TopicStatus = TopicStatus.IN_PROGRESS,
    difficulty: Difficulty | None = None,
    parent_id: str | None = None,
) -> Topic:
    """Seed a topic with domain derived from path's first segment."""
    domain = path.split(" > ", 1)[0]
    name = path.rsplit(" > ", 1)[-1]
    topic = Topic(
        path=path,
        domain=domain,
        name=name,
        status=status,
        difficulty=difficulty,
        parent_id=parent_id,
    )
    db.add(topic)
    db.flush()
    return topic


def _make_domain(
    db: DbSession,
    *,
    name: str,
    kind: DomainKind = DomainKind.LANGUAGE,
    description: str | None = None,
) -> Domain:
    """Seed a domain row."""
    domain = Domain(name=name, kind=kind, description=description)
    db.add(domain)
    db.flush()
    return domain


async def test_empty_db_returns_empty_lists(db: DbSession) -> None:
    """Empty database: both domains and topics arrive as empty lists."""
    response = await build_topics_response(db)

    assert response.domains == []
    assert response.topics == []


async def test_domains_sorted_alphabetically(db: DbSession) -> None:
    """Domain ordering is alphabetical by name."""
    _make_domain(db, name="Python")
    _make_domain(db, name="FastAPI", kind=DomainKind.FRAMEWORK)
    _make_domain(db, name="React", kind=DomainKind.LIBRARY)
    db.commit()

    response = await build_topics_response(db)

    names = [d.name for d in response.domains]
    assert names == ["FastAPI", "Python", "React"]


async def test_domain_summary_carries_kind_and_description(db: DbSession) -> None:
    """DomainSummary projects name, kind, description from the model."""
    _make_domain(
        db,
        name="Python",
        kind=DomainKind.LANGUAGE,
        description="A high-level programming language.",
    )
    db.commit()

    response = await build_topics_response(db)

    assert len(response.domains) == 1
    assert response.domains[0].name == "Python"
    assert response.domains[0].kind == DomainKind.LANGUAGE
    assert response.domains[0].description == "A high-level programming language."


async def test_topics_grouped_by_domain_then_path(db: DbSession) -> None:
    """Topics ordered by (domain, path) so each domain's topics arrive together."""
    _make_topic(db, path="Python > Functions > Closures")
    _make_topic(db, path="Python > Data Types > Integers")
    _make_topic(db, path="FastAPI > Routing > Path Parameters")
    _make_topic(db, path="React > Hooks > useState")
    db.commit()

    response = await build_topics_response(db)

    paths = [t.path for t in response.topics]
    assert paths == [
        "FastAPI > Routing > Path Parameters",
        "Python > Data Types > Integers",
        "Python > Functions > Closures",
        "React > Hooks > useState",
    ]


async def test_topics_include_all_statuses(db: DbSession) -> None:
    """Topics view returns topics in any status, not just in-progress."""
    _make_topic(db, path="Python > Started", status=TopicStatus.NOT_STARTED)
    _make_topic(db, path="Python > Working", status=TopicStatus.IN_PROGRESS)
    _make_topic(db, path="Python > Done", status=TopicStatus.LEARNED)
    _make_topic(db, path="Python > Rusty", status=TopicStatus.NEEDS_REVISION)
    db.commit()

    response = await build_topics_response(db)

    assert len(response.topics) == 4
    statuses = {t.status for t in response.topics}
    assert statuses == {
        TopicStatus.NOT_STARTED,
        TopicStatus.IN_PROGRESS,
        TopicStatus.LEARNED,
        TopicStatus.NEEDS_REVISION,
    }


async def test_topics_carry_difficulty_when_set(db: DbSession) -> None:
    """Topic difficulty is included in the summary and null is preserved."""
    _make_topic(
        db,
        path="Python > Data Types > Integers",
        difficulty=Difficulty.BEGINNER,
    )
    _make_topic(db, path="Python > Async", difficulty=None)
    db.commit()

    response = await build_topics_response(db)

    by_path = {t.path: t for t in response.topics}
    assert by_path["Python > Data Types > Integers"].difficulty == Difficulty.BEGINNER
    assert by_path["Python > Async"].difficulty is None


async def test_full_response_composes_domains_and_topics(db: DbSession) -> None:
    """End-to-end: seeded domains and topics both appear in the response."""
    _make_domain(db, name="Python", kind=DomainKind.LANGUAGE)
    _make_domain(db, name="FastAPI", kind=DomainKind.FRAMEWORK)
    _make_topic(db, path="Python > Data Types > Integers")
    _make_topic(db, path="FastAPI > Routing > Path Parameters")
    db.commit()

    response = await build_topics_response(db)

    assert len(response.domains) == 2
    assert len(response.topics) == 2
    domain_names = {d.name for d in response.domains}
    assert domain_names == {"FastAPI", "Python"}
    topic_paths = {t.path for t in response.topics}
    assert topic_paths == {
        "Python > Data Types > Integers",
        "FastAPI > Routing > Path Parameters",
    }
