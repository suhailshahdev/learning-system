"""Home dashboard service.

Composes the data the home screen needs into one response. Six
pieces: a blank-slate flag, the most recent in-progress session,
items due for review, in-progress topics grouped by domain,
recent sessions, and a knowledge summary by domain and difficulty.

Read-only with no transport calls or commits. The route layer is
a thin pass-through.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from sqlalchemy import func, select

from app.models import (
    LearnedItem,
    Session,
    SessionState,
    Topic,
    TopicStatus,
    UserKnowledgeAssertion,
    UserProfile,
)
from app.models.enums import Difficulty
from app.models.user_profile import SINGLETON_ID
from app.schemas.home import (
    DomainFocus,
    HomeResponse,
    KnowledgeSummaryRow,
    LearnedItemSummary,
    RecentSessionSummary,
    TopicSummary,
)

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession


# Maximum items in the "due for review" section. Conservative for a
# dashboard view; deeper review queues are a future surface.
DUE_FOR_REVIEW_LIMIT = 10

# Maximum in-progress topics shown per domain in focus_by_domain.
# Prevents one runaway domain from dominating the dashboard.
TOPICS_PER_DOMAIN_LIMIT = 10

# Maximum recent sessions in the dashboard.
RECENT_SESSIONS_LIMIT = 5


# Stable ordering for difficulty in the knowledge summary.
_DIFFICULTY_ORDER: dict[Difficulty, int] = {
    Difficulty.BEGINNER: 0,
    Difficulty.INTERMEDIATE: 1,
    Difficulty.ADVANCED: 2,
}


async def build_home_response(db: DbSession) -> HomeResponse:
    """Build the home dashboard payload.

    Runs six sub-queries against the existing schema. Returns a
    fully-populated HomeResponse the route layer can return as-is.
    """
    is_blank_slate = _is_blank_slate(db)
    continue_last = _continue_last(db)
    due_for_review = _due_for_review(db)
    focus_by_domain = _focus_by_domain(db)
    recent_sessions = _recent_sessions(db)
    knowledge_summary = _knowledge_summary(db)

    return HomeResponse(
        is_blank_slate=is_blank_slate,
        continue_last=continue_last,
        due_for_review=due_for_review,
        focus_by_domain=focus_by_domain,
        recent_sessions=recent_sessions,
        knowledge_summary=knowledge_summary,
    )


def _is_blank_slate(db: DbSession) -> bool:
    """True when the user has no profile config and no history.

    Three conditions must all hold: default_stack, resume_text, and
    target_jd_text are all unset or the profile row does not exist
    yet, plus zero learned items and zero sessions.
    """
    profile = db.get(UserProfile, SINGLETON_ID)
    if profile is not None and (
        profile.default_stack is not None
        or profile.resume_text is not None
        or profile.target_jd_text is not None
    ):
        return False

    item_count = db.scalar(select(func.count()).select_from(LearnedItem)) or 0
    if item_count > 0:
        return False

    session_count = db.scalar(select(func.count()).select_from(Session)) or 0
    return session_count == 0


def _continue_last(db: DbSession) -> RecentSessionSummary | None:
    """Most recent IN_PROGRESS session, or None.

    Multiple in-progress sessions are allowed by the schema.
    Returns the most recent one by created_at with topic_path
    joined for the home dashboard CTA. Topic-less sessions
    (cross-domain or pre-resolution) still resolve via outer
    join.
    """
    row = db.execute(
        select(Session, Topic.path)
        .join(Topic, Session.topic_id == Topic.id, isouter=True)
        .where(Session.state == SessionState.IN_PROGRESS)
        .order_by(Session.created_at.desc())
        .limit(1)
    ).one_or_none()

    if row is None:
        return None

    session, topic_path = row
    return RecentSessionSummary(
        id=session.id,
        topic_id=session.topic_id,
        topic_path=topic_path,
        state=session.state,
        transport_kind=session.transport_kind,
        mode_used=session.mode_used,
        created_at=session.created_at,
        updated_at=session.updated_at,
    )


def _due_for_review(db: DbSession) -> list[LearnedItemSummary]:
    """Up to DUE_FOR_REVIEW_LIMIT learned items, oldest review first.

    Joins LearnedItem to Topic to surface the topic path on the
    review card. Items with null last_reviewed_at sort last
    because null means "unknown" rather than "overdue."
    """
    rows = db.execute(
        select(LearnedItem, Topic.path)
        .join(Topic, LearnedItem.topic_id == Topic.id)
        .order_by(
            LearnedItem.last_reviewed_at.is_(None),
            LearnedItem.last_reviewed_at.asc(),
        )
        .limit(DUE_FOR_REVIEW_LIMIT)
    ).all()

    return [
        LearnedItemSummary(
            id=item.id,
            question=item.question,
            topic_path=topic_path,
            difficulty=item.difficulty,
            mode=item.mode,
            last_reviewed_at=item.last_reviewed_at,
        )
        for item, topic_path in rows
    ]


def _focus_by_domain(db: DbSession) -> list[DomainFocus]:
    """In-progress topics grouped by domain.

    One DomainFocus entry per domain that has at least one
    in-progress topic. Topics within each domain are capped at
    TOPICS_PER_DOMAIN_LIMIT to keep the dashboard render bounded.
    """
    topics = (
        db.execute(
            select(Topic)
            .where(Topic.status == TopicStatus.IN_PROGRESS)
            .order_by(Topic.domain.asc(), Topic.path.asc())
        )
        .scalars()
        .all()
    )

    grouped: dict[str, list[TopicSummary]] = defaultdict(list)
    for topic in topics:
        if len(grouped[topic.domain]) >= TOPICS_PER_DOMAIN_LIMIT:
            continue
        grouped[topic.domain].append(TopicSummary.model_validate(topic))

    return [
        DomainFocus(domain=domain, in_progress_topics=topics_in_domain)
        for domain, topics_in_domain in sorted(grouped.items())
    ]


def _recent_sessions(db: DbSession) -> list[RecentSessionSummary]:
    """Last RECENT_SESSIONS_LIMIT sessions, most recent first.

    Left-joins Topic so the row carries topic_path for dashboard
    rendering. topic_path is None when the session has no topic
    (cross-domain sessions or sessions abandoned before topic
    resolution).
    """
    rows = db.execute(
        select(Session, Topic.path)
        .join(Topic, Session.topic_id == Topic.id, isouter=True)
        .order_by(Session.created_at.desc())
        .limit(RECENT_SESSIONS_LIMIT)
    ).all()

    return [
        RecentSessionSummary(
            id=session.id,
            topic_id=session.topic_id,
            topic_path=topic_path,
            state=session.state,
            transport_kind=session.transport_kind,
            mode_used=session.mode_used,
            created_at=session.created_at,
            updated_at=session.updated_at,
        )
        for session, topic_path in rows
    ]


def _knowledge_summary(db: DbSession) -> list[KnowledgeSummaryRow]:
    """Aggregate user knowledge assertions by (domain, difficulty).

    Domain is derived from topic_path's first segment. Counts
    distinct topic_paths per (domain, difficulty) so the same
    topic asserted by multiple sources counts as one.
    """
    assertions = db.execute(select(UserKnowledgeAssertion)).scalars().all()

    # (domain, difficulty) -> set of distinct topic_paths
    buckets: dict[tuple[str, Difficulty], set[str]] = defaultdict(set)
    for assertion in assertions:
        domain = assertion.topic_path.split(" > ", 1)[0]
        buckets[(domain, assertion.difficulty)].add(assertion.topic_path)

    rows = [
        KnowledgeSummaryRow(domain=domain, difficulty=difficulty, count=len(paths))
        for (domain, difficulty), paths in buckets.items()
    ]
    rows.sort(key=lambda r: (r.domain, _DIFFICULTY_ORDER[r.difficulty]))
    return rows
