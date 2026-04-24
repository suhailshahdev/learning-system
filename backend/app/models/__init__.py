"""SQLAlchemy models for the learning system.

Every model registers against `Base` and should mix in `UUIDPrimaryKey`
and `Timestamps` unless there is a specific reason not to.
"""

from app.models.base import Base, Timestamps, UUIDPrimaryKey
from app.models.domain import Domain
from app.models.enums import AssertionSource, Difficulty, DomainKind
from app.models.teaching_preference import TeachingPreference
from app.models.user_knowledge_assertion import UserKnowledgeAssertion
from app.models.user_profile import SINGLETON_ID, UserProfile

__all__ = [
    "SINGLETON_ID",
    "AssertionSource",
    "Base",
    "Difficulty",
    "Domain",
    "DomainKind",
    "TeachingPreference",
    "Timestamps",
    "UUIDPrimaryKey",
    "UserKnowledgeAssertion",
    "UserProfile",
]
