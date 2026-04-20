"""Reusable FastAPI dependency aliases.

Annotated types here let route handlers declare common dependencies
(database session, future auth, etc.) without repeating the Depends()
boilerplate. Using Annotated rather than Depends-in-default keeps the
call out of argument defaults (ruff B008) and matches FastAPI's
recommended style.
"""

from typing import Annotated

from fastapi import Depends
from sqlalchemy.orm import Session

from app.core.db import get_db

DbSession = Annotated[Session, Depends(get_db)]
"""A per-request SQLAlchemy session, closed automatically on return."""
