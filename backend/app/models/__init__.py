"""SQLAlchemy models for the learning system.

Every model registers against `Base` and should mix in `UUIDPrimaryKey`
and `Timestamps` unless there is a specific reason not to.
"""

from app.models.base import Base, Timestamps, UUIDPrimaryKey
from app.models.domain import Domain
from app.models.enums import (
    AssertionSource,
    Difficulty,
    DomainKind,
    LearnedItemStatus,
    LearningMode,
    SessionState,
    TopicStatus,
    TransportKind,
    TurnRole,
)
from app.models.error_log import ErrorLog
from app.models.learned_item import LearnedItem
from app.models.session import Session
from app.models.session_turn import SessionTurn
from app.models.teaching_preference import TeachingPreference
from app.models.topic import Topic
from app.models.user_knowledge_assertion import UserKnowledgeAssertion
from app.models.user_profile import SINGLETON_ID, UserProfile

__all__ = [
    "SINGLETON_ID",
    "AssertionSource",
    "Base",
    "Difficulty",
    "Domain",
    "DomainKind",
    "ErrorLog",
    "LearnedItem",
    "LearnedItemStatus",
    "LearningMode",
    "Session",
    "SessionState",
    "SessionTurn",
    "TeachingPreference",
    "Timestamps",
    "Topic",
    "TopicStatus",
    "TransportKind",
    "TurnRole",
    "UUIDPrimaryKey",
    "UserKnowledgeAssertion",
    "UserProfile",
]
