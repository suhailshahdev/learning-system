"""FastAPI application entry point.

This module offers two ways to get the app: a factory function
(`create_app`) used by tests, and a module-level `app` object
that Uvicorn imports at startup. The factory is the real builder.
The module-level `app` exists only because ASGI servers need an
import path to point at.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import health
from app.core.config import Settings, get_settings


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
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=resolved.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/api")

    return app


app = create_app()
