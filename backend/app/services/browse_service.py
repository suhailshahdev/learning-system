"""Sessions browse service.

One read function: list_sessions returns up to BROWSE_LIMIT
sessions, optionally filtered by state, sorted by created_at desc.
Each row carries a learned_item_count so the browse page can
distinguish substantive sessions from empty ones.

Read-only with no transport calls or commits.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import func, select

from app.models import LearnedItem, Session, SessionState, Topic
from app.schemas.browse_api import BrowseResponse, BrowseSessionRow

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession


# Maximum rows returned by the browse endpoint. A single-user
# local app accumulating 50 sessions is months of usage. The cap
# defers proper pagination until real friction surfaces. When the
# cap is hit, BrowseResponse.limit_reached is True so the frontend
# can prompt the user accordingly.
BROWSE_LIMIT = 50


def list_sessions(
    db: DbSession,
    *,
    state: SessionState | None = None,
) -> BrowseResponse:
    """Return browse rows for all sessions, optionally state-filtered.

    state=None returns all states. Passing one of the four
    SessionState values filters to that state. Rows are sorted
    by created_at desc.

    learned_item_count joins LearnedItem via a correlated subquery
    rather than a GROUP BY on the main query because:
    - SQLite's GROUP BY semantics on aggregates with non-aggregate
      columns is fragile across versions
    - The browse page doesn't sort by count today, so the join
      doesn't need to drive ordering
    - At BROWSE_LIMIT=50 the count subquery runs 50 times, which
      is fine for a local app with a few hundred items
    """
    item_count_subquery = (
        select(func.count(LearnedItem.id))
        .where(LearnedItem.session_id == Session.id)
        .correlate(Session)
        .scalar_subquery()
    )

    stmt = (
        select(Session, Topic.path, item_count_subquery.label("item_count"))
        .join(Topic, Session.topic_id == Topic.id, isouter=True)
        .order_by(Session.created_at.desc())
    )
    if state is not None:
        stmt = stmt.where(Session.state == state)
    stmt = stmt.limit(BROWSE_LIMIT + 1)

    raw_rows = db.execute(stmt).all()

    # The limit + 1 trick: if BROWSE_LIMIT + 1 rows came back,
    # there is at least one more session past the cap. Trim to
    # BROWSE_LIMIT in the response, set limit_reached=True.
    limit_reached = len(raw_rows) > BROWSE_LIMIT
    visible_rows = raw_rows[:BROWSE_LIMIT]

    rows = [
        BrowseSessionRow(
            id=session.id,
            topic_id=session.topic_id,
            topic_path=topic_path,
            state=session.state,
            transport_kind=session.transport_kind,
            mode_used=session.mode_used,
            learned_item_count=item_count,
            created_at=session.created_at,
            updated_at=session.updated_at,
        )
        for session, topic_path, item_count in visible_rows
    ]
    return BrowseResponse(rows=rows, limit_reached=limit_reached)
