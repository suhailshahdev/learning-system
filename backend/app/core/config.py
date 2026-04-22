"""Application configuration loaded from environment variables

Settings are validated at startup. Missing required values raise on first access,
which is what we want: fail loudly and early rather than silently using wrong values.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "test", "production"]


class Settings(BaseSettings):
    """Typed configuration, loaded from the process environment or a local .env file.

    Fields are documented inline. To add a new setting:
        1. Add the field here with a type and a default (or no default if required).
        2. Add the variable to .env.example with a comment.
        3. Reference it through get_settings() in the code that needs it.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="forbid",
    )

    environment: Environment = Field(
        default="local",
        description="Deployment environment. Controls debug behaviours.",
    )

    database_url: str = Field(description="SQLAlchemy connection string for the primary database.")

    cors_allow_origins: list[str] = Field(
        default=["http://localhost:5173"],
        description=(
            "Origins permitted to call the API. The Vite dev server runs on "
            "5173; production deployments (if ever) should override this."
        ),
    )

    @property
    def is_local(self) -> bool:
        """True when running in a local development context."""
        return self.environment == "local"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the application's settings singleton.

    Cached so the .env file is read once per process. Use this in dependency
    injection rather than constructing Settings() directly; that keeps tests
    able to override it.
    """
    return Settings()
