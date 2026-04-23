"""SQLAlchemy models for the learning system.

Every model registers against `Base` and should mix in `UUIDPrimaryKey`
and `Timestamps` unless there is a specific reason not to.
"""

from app.models.base import Base, Timestamps, UUIDPrimaryKey

__all__ = ["Base", "Timestamps", "UUIDPrimaryKey"]
