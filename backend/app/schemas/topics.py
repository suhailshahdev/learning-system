"""Response schemas for the topics view endpoint.

The topics view composes domains and topics into one response.
Domains come from the seeded reference table and topics come from
the live tree built as sessions run. The frontend assembles the
nested tree from these two flat lists.
"""

from __future__ import annotations

# Pydantic v2 resolves field type annotations at validation time.
# Generic-wrapped types (list[X]) require runtime imports.
from app.models.enums import DomainKind  # noqa: TC002
from app.schemas.home import TopicSummary  # noqa: TC002
from pydantic import BaseModel, ConfigDict


class DomainSummary(BaseModel):
    """Compact projection of a Domain row.

    Carries the fields the topics view needs to render a domain
    header: name (the path segment topics use), kind (so the UI
    can group or label by kind if it wants), description (free
    text the user might add later).
    """

    model_config = ConfigDict(frozen=True, from_attributes=True)

    name: str
    kind: DomainKind
    description: str | None


class TopicsResponse(BaseModel):
    """Composed payload for GET /api/topics.

    Two flat lists. The frontend builds the nested tree by
    grouping topics under domains (Topic.domain matches
    DomainSummary.name) and using parent_id for nesting within
    a domain.
    """

    model_config = ConfigDict(frozen=True)

    domains: list[DomainSummary]
    topics: list[TopicSummary]
