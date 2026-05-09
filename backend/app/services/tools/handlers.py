"""Tool handlers for the LLM tool surface.

Each handler takes a database session plus a validated input
schema and returns a validated output schema. Handlers are
async to match the transport interface and to allow future
handlers that call other services without changing the
signature.

Handlers commit their own writes. A failure to produce a valid
teaching turn after a successful tool call should not roll back
the tool call.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.models import (
    Domain,
    Session,
    Topic,
    UserKnowledgeAssertion,
)
from app.models.enums import Difficulty
from app.schemas.common import Prerequisite
from app.schemas.tools import (
    CreateDomainInput,
    CreateDomainOutput,
    CreateOrUpdateTopicInput,
    CreateOrUpdateTopicOutput,
    DomainInfo,
    GetRecentSessionsInput,
    GetRecentSessionsOutput,
    GetTopicsByDomainInput,
    GetTopicsByDomainOutput,
    GetUserKnowledgeSummaryInput,
    GetUserKnowledgeSummaryOutput,
    KnowledgeRow,
    ListDomainsInput,
    ListDomainsOutput,
    RecentSessionInfo,
    TopicInfo,
)
from app.services.topic_crud import get_or_create_topic

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession


# Stable ordering for difficulty in the knowledge summary.
# Matches the home dashboard's ordering so the LLM and the user
# see the same shape.
_DIFFICULTY_ORDER: dict[Difficulty, int] = {
    Difficulty.BEGINNER: 0,
    Difficulty.INTERMEDIATE: 1,
    Difficulty.ADVANCED: 2,
}


class ToolHandlerError(Exception):
    """A tool handler failed.

    Wrapped errors that the session-service loop can format back
    to the LLM as a tool result. The LLM sees the message and
    can decide to retry or proceed differently.
    """

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.cause = cause


async def list_domains(db: DbSession, args: ListDomainsInput) -> ListDomainsOutput:
    """Return all domains, alphabetical by name.

    Read-only, no commit needed.
    """
    domains = db.execute(select(Domain).order_by(Domain.name.asc())).scalars().all()
    return ListDomainsOutput(
        domains=[DomainInfo(name=d.name, kind=d.kind, description=d.description) for d in domains]
    )


async def create_domain(db: DbSession, args: CreateDomainInput) -> CreateDomainOutput:
    """Insert a new domain or return the existing one.

    Idempotent on name. The LLM may call this whenever it
    encounters a domain that's not in the inventory. If the
    domain already exists, the existing row is returned with
    `created=False`.
    """
    existing = db.execute(select(Domain).where(Domain.name == args.name)).scalar_one_or_none()
    if existing is not None:
        return CreateDomainOutput(
            created=False,
            domain=DomainInfo(
                name=existing.name, kind=existing.kind, description=existing.description
            ),
        )

    new_domain = Domain(name=args.name, kind=args.kind, description=args.description)
    db.add(new_domain)
    try:
        db.commit()
    except SQLAlchemyError as e:
        db.rollback()
        raise ToolHandlerError(f"Failed to create domain {args.name!r}: {e}", cause=e) from e
    db.refresh(new_domain)

    return CreateDomainOutput(
        created=True,
        domain=DomainInfo(
            name=new_domain.name, kind=new_domain.kind, description=new_domain.description
        ),
    )


async def get_topics_by_domain(
    db: DbSession, args: GetTopicsByDomainInput
) -> GetTopicsByDomainOutput:
    """Return existing topics within one domain.

    Filtered by domain name and ordered by path. Returns an empty
    list if the domain has no topics yet or does not exist as a
    domain row, since Topic.domain is denormalized and can hold
    values not in the domain table.
    """
    topics = (
        db.execute(select(Topic).where(Topic.domain == args.domain_name).order_by(Topic.path.asc()))
        .scalars()
        .all()
    )

    return GetTopicsByDomainOutput(
        domain=args.domain_name,
        topics=[_topic_info(t) for t in topics],
    )


async def create_or_update_topic(
    db: DbSession, args: CreateOrUpdateTopicInput
) -> CreateOrUpdateTopicOutput:
    """Upsert a topic by path with optional metadata.

    Calls topic_crud.get_or_create_topic for the upsert-by-path
    semantics, then optionally updates difficulty, prerequisites,
    and parent_id from the args.

    None-valued fields leave existing values unchanged. The LLM
    can call this with just `path` to ensure a topic exists, or
    with full metadata to set everything at once.
    """

    existing_paths = db.execute(select(Topic.path).where(Topic.path == args.path)).scalar()
    is_new = existing_paths is None

    topic = get_or_create_topic(db, args.path)

    if args.difficulty is not None:
        topic.difficulty = args.difficulty

    if args.prerequisites:
        topic.prerequisites = [p.model_dump(mode="json") for p in args.prerequisites]

    if args.parent_path is not None:
        parent = db.execute(
            select(Topic).where(Topic.path == args.parent_path)
        ).scalar_one_or_none()
        if parent is None:
            raise ToolHandlerError(
                f"parent_path {args.parent_path!r} does not exist; create it first."
            )
        topic.parent_id = parent.id

    try:
        db.commit()
    except SQLAlchemyError as e:
        db.rollback()
        raise ToolHandlerError(f"Failed to upsert topic {args.path!r}: {e}", cause=e) from e
    db.refresh(topic)

    return CreateOrUpdateTopicOutput(created=is_new, topic=_topic_info(topic))


async def get_user_knowledge_summary(
    db: DbSession, args: GetUserKnowledgeSummaryInput
) -> GetUserKnowledgeSummaryOutput:
    """Aggregate user knowledge assertions by (domain, difficulty).

    Same query as home_service._knowledge_summary. Domain comes
    from the first segment of topic_path, counts distinct topic
    paths per (domain, difficulty) so the same topic asserted by
    multiple sources counts once.
    """
    assertions = db.execute(select(UserKnowledgeAssertion)).scalars().all()

    buckets: dict[tuple[str, Difficulty], set[str]] = defaultdict(set)
    for assertion in assertions:
        domain = assertion.topic_path.split(" > ", 1)[0]
        buckets[(domain, assertion.difficulty)].add(assertion.topic_path)

    rows = [
        KnowledgeRow(domain=domain, difficulty=difficulty, count=len(paths))
        for (domain, difficulty), paths in buckets.items()
    ]
    rows.sort(key=lambda r: (r.domain, _DIFFICULTY_ORDER[r.difficulty]))
    return GetUserKnowledgeSummaryOutput(rows=rows)


async def get_recent_sessions(
    db: DbSession, args: GetRecentSessionsInput
) -> GetRecentSessionsOutput:
    """Return the last N sessions, most recent first.

    Slimmer projection than the home dashboard's RecentSessionSummary:
    the LLM doesn't need session id, transport kind, or updated_at.
    Topic_path is the most useful field for context.
    """
    rows = db.execute(
        select(Session, Topic.path)
        .join(Topic, Session.topic_id == Topic.id, isouter=True)
        .order_by(Session.created_at.desc())
        .limit(args.limit)
    ).all()

    return GetRecentSessionsOutput(
        sessions=[
            RecentSessionInfo(
                topic_path=topic_path,
                state=session.state,
                mode_used=session.mode_used,
                created_at=session.created_at,
            )
            for session, topic_path in rows
        ]
    )


def _topic_info(topic: Topic) -> TopicInfo:
    """Project a Topic row into the slimmer TopicInfo tool shape."""
    return TopicInfo(
        path=topic.path,
        difficulty=topic.difficulty,
        status=topic.status,
        prerequisites=[Prerequisite.model_validate(p) for p in topic.prerequisites],
    )
