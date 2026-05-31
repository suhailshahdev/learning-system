"""FastAPI application entry point.

This module offers two ways to get the app: a factory function
(`create_app`) used by tests, and a module-level `app` object
that Uvicorn imports at startup. The factory is the real builder.
The module-level `app` exists only because ASGI servers need an
import path to point at.
"""

from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import admin, diagnose, documents, health, home, search, sessions, topics
from app.core.config import Settings, get_settings
from app.core.db import SessionLocal
from app.core.telemetry import configure_tracing
from app.models import TransportKind
from app.services.embedding_service import OpenRouterEmbedder
from app.services.llm_call_recorder import WritingRecorder
from app.transport.deepseek_impl import DeepseekTransport
from app.transport.instrumented import InstrumentedTransport
from app.transport.playwright_impl import PlaywrightClaudeTransport


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Construct and tear down app-scoped resources.

    Both LLM transports are constructed once at startup and held
    for the lifetime of the process. Playwright pays a 5-10 second
    browser-launch cost here so per-request handling stays fast.
    DeepSeek follows the same pattern for symmetry; its startup
    is cheap.

    If either transport fails to start (logged-out claude.ai
    profile, missing DeepSeek key) the app fails to start.
    """
    settings: Settings = app.state.settings
    if settings.enable_tracing:
        configure_tracing()
    recorder = WritingRecorder(SessionLocal)
    async with AsyncExitStack() as stack:
        playwright_transport = await stack.enter_async_context(
            PlaywrightClaudeTransport(settings.chrome_profile_path)
        )
        deepseek_transport = await stack.enter_async_context(
            DeepseekTransport(
                api_key=settings.deepseek_api_key.get_secret_value(),
                default_model=settings.deepseek_model,
            )
        )
        embedder = await stack.enter_async_context(
            OpenRouterEmbedder(
                api_key=settings.openrouter_api_key.get_secret_value(),
                model=settings.openrouter_embedding_model,
            )
        )
        # Wrap each transport so every round-trip is recorded. The
        # wrapper cannot see transport_kind or model through the
        # Protocol, so they are supplied here where the concrete
        # transport is known. claude.ai exposes no model id.
        app.state.playwright_transport = InstrumentedTransport(
            playwright_transport,
            recorder,
            TransportKind.CLAUDE_PLAYWRIGHT,
            model=None,
        )
        app.state.deepseek_transport = InstrumentedTransport(
            deepseek_transport,
            recorder,
            TransportKind.DEEPSEEK,
            model=settings.deepseek_model,
        )
        app.state.embedder = embedder
        yield


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and return a configured FastAPI application

    Accepts an optional Settings instance so tests can inject their
    own configuration (test database, custom CORS origin, etc.)
    without touching the global settings cache
    """
    resolved = settings or get_settings()
    app = FastAPI(
        title="Learning System API",
        version="0.1.0",
        description="Local backend for the personal learning system.",
        lifespan=lifespan,
    )
    app.state.settings = resolved
    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(admin.router, prefix="/api")
    app.include_router(diagnose.router, prefix="/api")
    app.include_router(documents.router, prefix="/api")
    app.include_router(health.router, prefix="/api")
    app.include_router(home.router, prefix="/api")
    app.include_router(search.router, prefix="/api")
    app.include_router(sessions.router, prefix="/api")
    app.include_router(topics.router, prefix="/api")
    return app


app = create_app()
