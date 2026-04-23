"""SQLAlchemy models for the learning system.

Every model registers against `Base` and should mix in `UUIDPrimaryKey`
and `Timestamps` unless there is a specific reason not to.
"""

from app.models.base import Base, Timestamps, UUIDPrimaryKey
from app.models.domain import Domain
from app.models.enums import DomainKind
from app.models.teaching_preference import TeachingPreference

__all__ = [
    "Base",
    "Domain",
    "DomainKind",
    "TeachingPreference",
    "Timestamps",
    "UUIDPrimaryKey",
]
