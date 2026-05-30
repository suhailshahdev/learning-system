"""SQLAlchemy models for the learning system.

Every model registers against `Base` and should mix in `UUIDPrimaryKey`
and `Timestamps` unless there is a specific reason not to.
"""

from app.models.base import Base, Timestamps, UUIDPrimaryKey
from app.models.document import Document
from app.models.domain import Domain
from app.models.embedding import Embedding
from app.models.enums import (
    AssertionSource,
    Difficulty,
    DomainKind,
    EmbeddingSourceType,
    GradingVerdict,
    LearnedItemStatus,
    LearningMode,
    SessionState,
    TopicStatus,
    TransportKind,
    TurnRole,
)
from app.models.error_log import ErrorLog
from app.models.learned_item import LearnedItem
from app.models.llm_call import LLMCall
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
    "Document",
    "Domain",
    "DomainKind",
    "Embedding",
    "EmbeddingSourceType",
    "ErrorLog",
    "GradingVerdict",
    "LLMCall",
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
