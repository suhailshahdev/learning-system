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
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.models import (
    Domain,
    LearnedItem,
    Session,
    Topic,
    UserKnowledgeAssertion,
)
from app.models.enums import Difficulty, GradingVerdict
from app.schemas.common import Prerequisite
from app.schemas.tools import (
    CreateDomainInput,
    CreateDomainOutput,
    CreateOrUpdateTopicInput,
    CreateOrUpdateTopicOutput,
    DomainInfo,
    GetRecentSessionsInput,
    GetRecentSessionsOutput,
    GetStaleTopicsInput,
    GetStaleTopicsOutput,
    GetTopicsByDomainInput,
    GetTopicsByDomainOutput,
    GetUserKnowledgeSummaryInput,
    GetUserKnowledgeSummaryOutput,
    GetWeakTopicsInput,
    GetWeakTopicsOutput,
    KnowledgeRow,
    ListDomainsInput,
    ListDomainsOutput,
    RecentSessionInfo,
    StaleTopicInfo,
    TopicInfo,
    WeakTopicInfo,
    WrongAnswerSample,
)
from app.services.topic_crud import get_or_create_topic_with_ancestors

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

    Walks the leaf path so any missing ancestors are auto-created
    with minimal metadata. If parent_path is supplied, that chain
    is walked too and the leaf's parent_id is wired to it.

    None-valued fields leave existing values unchanged on the leaf.
    The LLM can call this with just `path` to ensure a topic exists,
    or with full metadata to set everything at once.

    Auto-created ancestors get IN_PROGRESS status and NULL difficulty.
    The LLM can fill metadata in later by calling this handler again
    with that ancestor as the leaf path.
    """

    existing_paths = db.execute(select(Topic.path).where(Topic.path == args.path)).scalar()
    is_new = existing_paths is None

    topic = get_or_create_topic_with_ancestors(db, args.path)

    if args.difficulty is not None:
        topic.difficulty = args.difficulty

    if args.prerequisites:
        topic.prerequisites = [p.model_dump(mode="json") for p in args.prerequisites]

    if args.parent_path is not None:
        parent = get_or_create_topic_with_ancestors(db, args.parent_path)
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


# Weakness score weighting. Incorrect counts as 1.0, partial as 0.5,
# correct as 0.0. Pulled into a constant so the formula is named
# and a future tweak (e.g. weighing partial higher because it means
# the user almost had it) has one place to land.
_INCORRECT_WEIGHT = 1.0
_PARTIAL_WEIGHT = 0.5


async def get_weak_topics(db: DbSession, args: GetWeakTopicsInput) -> GetWeakTopicsOutput:
    """Aggregate learned items by topic, surface topics with weakness.

    Walks every LearnedItem that has a grading_verdict, groups by
    topic, counts verdicts, and returns topics whose total attempt
    count meets min_attempts and whose weakness score is non-zero.
    Topics where every attempt was correct are skipped (nothing to
    diagnose).

    Ordered worst-first by weakness score. Up to sample_size
    representative wrong-answer questions are attached per topic,
    truncated at 200 chars. Setting sample_size=0
    returns counts only.

    Read-only, no commit needed.
    """
    items = db.execute(
        select(LearnedItem, Topic.path)
        .join(Topic, LearnedItem.topic_id == Topic.id)
        .where(LearnedItem.grading_verdict.is_not(None))
        .order_by(LearnedItem.last_reviewed_at.desc())
    ).all()

    # Group items by topic_path. Each group accumulates verdict counts
    # and a bounded queue of sample questions (most recent first via
    # the query's ORDER BY).
    counts_by_path: dict[str, dict[GradingVerdict, int]] = defaultdict(
        lambda: dict.fromkeys(GradingVerdict, 0)
    )
    samples_by_path: dict[str, list[WrongAnswerSample]] = defaultdict(list)

    for item, topic_path in items:
        verdict = item.grading_verdict
        counts_by_path[topic_path][verdict] += 1

        if (
            args.sample_size > 0
            and verdict in (GradingVerdict.INCORRECT, GradingVerdict.PARTIAL)
            and len(samples_by_path[topic_path]) < args.sample_size
        ):
            samples_by_path[topic_path].append(
                WrongAnswerSample(
                    question=_truncate(item.question, 200),
                    verdict=verdict,
                )
            )

    weak_topics: list[tuple[float, WeakTopicInfo]] = []
    for topic_path, counts in counts_by_path.items():
        total = sum(counts.values())
        if total < args.min_attempts:
            continue

        incorrect = counts[GradingVerdict.INCORRECT]
        partial = counts[GradingVerdict.PARTIAL]
        correct = counts[GradingVerdict.CORRECT]

        score = (incorrect * _INCORRECT_WEIGHT + partial * _PARTIAL_WEIGHT) / total
        if score == 0.0:
            continue

        weak_topics.append(
            (
                score,
                WeakTopicInfo(
                    topic_path=topic_path,
                    incorrect_count=incorrect,
                    partial_count=partial,
                    correct_count=correct,
                    samples=samples_by_path[topic_path],
                ),
            )
        )

    # Worst-first ordering. Tie-break by topic_path so the result is
    # deterministic when two topics have identical scores.
    weak_topics.sort(key=lambda pair: (-pair[0], pair[1].topic_path))

    return GetWeakTopicsOutput(topics=[info for _, info in weak_topics])


async def get_stale_topics(db: DbSession, args: GetStaleTopicsInput) -> GetStaleTopicsOutput:
    """Surface topics with old last_reviewed_at timestamps.

    Reads every Topic with a last_reviewed_at older than the
    threshold and returns it. Topics that have never been reviewed
    (last_reviewed_at IS NULL) are skipped: "never reviewed" is a
    different signal from "haven't revisited in a while" and the
    diagnostic LLM has list_domains and the knowledge summary for
    that case already.

    Ordered oldest-first. Limit caps the result so the LLM does not
    get a flood when the user has many stale topics.

    Read-only, no commit needed.
    """
    now = datetime.now(UTC)
    threshold = now - timedelta(days=args.days_threshold)

    topics = (
        db.execute(
            select(Topic)
            .where(Topic.last_reviewed_at.is_not(None))
            .where(Topic.last_reviewed_at < threshold)
            .order_by(Topic.last_reviewed_at.asc())
            .limit(args.limit)
        )
        .scalars()
        .all()
    )

    result: list[StaleTopicInfo] = []
    for topic in topics:
        # Query filters last_reviewed_at IS NOT NULL but mypy cannot
        # carry that proof to the call site. Assert documents the
        # query-level invariant and narrows the type.
        assert topic.last_reviewed_at is not None
        reviewed_at = _aware(topic.last_reviewed_at)
        result.append(
            StaleTopicInfo(
                topic_path=topic.path,
                last_reviewed_at=reviewed_at,
                days_since_review=(now - reviewed_at).days,
            )
        )
    return GetStaleTopicsOutput(topics=result)


def _truncate(text: str, max_chars: int) -> str:
    """Cap text at max_chars, appending an ellipsis when truncated.

    Used by get_weak_topics to enforce 200-char cap on sample
    questions. The ellipsis takes one of those chars rather than
    being appended after, so the cap is honored.
    """
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def _aware(dt: datetime) -> datetime:
    """Return a timezone-aware datetime, assuming UTC for naive inputs.

    SQLite drops tz info on DateTime(timezone=True) columns, so
    last_reviewed_at comes back naive even though it was written aware.
    This helper normalizes to UTC for the subtraction in
    get_stale_topics so timedelta math is consistent.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def _topic_info(topic: Topic) -> TopicInfo:
    """Project a Topic row into the slimmer TopicInfo tool shape."""
    return TopicInfo(
        path=topic.path,
        difficulty=topic.difficulty,
        status=topic.status,
        prerequisites=[Prerequisite.model_validate(p) for p in topic.prerequisites],
    )
