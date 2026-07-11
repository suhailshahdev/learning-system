"""Reusable FastAPI dependency aliases.

Annotated types here let route handlers declare common dependencies
(database session, transports, future auth) without repeating the
Depends() boilerplate. Using Annotated rather than Depends-in-default
keeps the call out of argument defaults (ruff B008) and matches
FastAPI's recommended style.
"""

from typing import Annotated, Any

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from app.core.db import SessionLocal, get_db
from app.models import TransportKind
from app.services.agent_error_recorder import WritingAgentErrorRecorder
from app.services.embedding_service import OpenRouterEmbedder
from app.transport.base import LLMTransport
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


def get_embedder(request: Request) -> OpenRouterEmbedder:
    """Return the app-scoped embedder from app.state."""
    return request.app.state.embedder  # type: ignore[no-any-return]


def get_agent_error_recorder() -> WritingAgentErrorRecorder:
    """Return a recorder that writes agent errors on its own sessions.

    Constructed per request but stateless: it holds the SessionLocal
    factory, not a live session, so construction costs nothing and
    each write opens its own short session independent of the
    request's transaction.
    """
    return WritingAgentErrorRecorder(SessionLocal)


PlaywrightTransportDep = Annotated[PlaywrightClaudeTransport, Depends(get_playwright_transport)]
"""The app-scoped Playwright + claude.ai transport."""

DeepseekTransportDep = Annotated[DeepseekTransport, Depends(get_deepseek_transport)]
"""The app-scoped DeepSeek chat completions transport."""

EmbedderDep = Annotated[OpenRouterEmbedder, Depends(get_embedder)]
"""The app-scoped OpenRouter embedder."""

AgentErrorRecorderDep = Annotated[WritingAgentErrorRecorder, Depends(get_agent_error_recorder)]
"""A writing agent-error recorder over the app's session factory."""


def pick_transport(
    kind: TransportKind,
    playwright: PlaywrightClaudeTransport,
    deepseek: DeepseekTransport,
) -> LLMTransport[Any]:
    """Dispatch to the matching transport instance.

    Lived as a per-module copy in sessions.py and diagnose.py until
    the agent routes became the third caller. Both transports are
    constructed at app startup and held on app.state; the route reads
    the kind from the request body, this function picks.

    Returns LLMTransport[Any] rather than LLMTransport[object]
    because Handle is invariant: a PlaywrightClaudeTransport is
    LLMTransport[PlaywrightChatHandle], not LLMTransport[object].
    Any matches the convention used by the service signatures.
    """
    if kind is TransportKind.CLAUDE_PLAYWRIGHT:
        return playwright
    return deepseek
