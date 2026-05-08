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

from app.api import health, home, sessions, topics
from app.core.config import Settings, get_settings
from app.transport.deepseek_impl import DeepseekTransport
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
        app.state.playwright_transport = playwright_transport
        app.state.deepseek_transport = deepseek_transport
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
    app.include_router(health.router, prefix="/api")
    app.include_router(home.router, prefix="/api")
    app.include_router(sessions.router, prefix="/api")
    app.include_router(topics.router, prefix="/api")
    return app


app = create_app()
