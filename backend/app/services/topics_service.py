"""Topics view service.

Composes the domains reference list and the topic tree into a
single response. Read-only with no transport calls or commits.
The route layer is a thin pass-through.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import select

from app.models import Domain, Topic
from app.schemas.home import TopicSummary
from app.schemas.topics import DomainSummary, TopicsResponse

if TYPE_CHECKING:
    from sqlalchemy.orm import Session as DbSession


async def build_topics_response(db: DbSession) -> TopicsResponse:
    """Build the topics view payload.

    Returns all domains (alphabetical) and all topics (by domain
    then path so each domain's topics arrive in stable order
    regardless of insert order). Tree assembly happens on the
    frontend.
    """
    domains = db.execute(select(Domain).order_by(Domain.name.asc())).scalars().all()
    topics = (
        db.execute(select(Topic).order_by(Topic.domain.asc(), Topic.path.asc())).scalars().all()
    )

    return TopicsResponse(
        domains=[DomainSummary.model_validate(d) for d in domains],
        topics=[TopicSummary.model_validate(t) for t in topics],
    )
