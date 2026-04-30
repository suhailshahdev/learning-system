"""Reusable FastAPI dependency aliases.

Annotated types here let route handlers declare common dependencies
(database session, transports, future auth) without repeating the
Depends() boilerplate. Using Annotated rather than Depends-in-default
keeps the call out of argument defaults (ruff B008) and matches
FastAPI's recommended style.
"""

from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.transport.deepseek_impl import DeepseekTransport
from app.transport.playwright_impl import PlaywrightClaudeTransport

DbSession = Annotated[Session, Depends(get_db)]
"""A per-request SQLAlchemy session, closed automatically on return."""


def get_playwright_transport(request: Request) -> PlaywrightClaudeTransport:
    """Return the app-scoped Playwright transport from app.state."""
    return request.app.state.playwright_transport  # type: ignore[no-any-return]


def get_deepseek_transport(request: Request) -> DeepseekTransport:
    """Return the app-scoped DeepSeek transport from app.state."""
    return request.app.state.deepseek_transport  # type: ignore[no-any-return]


PlaywrightTransportDep = Annotated[PlaywrightClaudeTransport, Depends(get_playwright_transport)]
"""The app-scoped Playwright + claude.ai transport."""

DeepseekTransportDep = Annotated[DeepseekTransport, Depends(get_deepseek_transport)]
"""The app-scoped DeepSeek chat completions transport."""
