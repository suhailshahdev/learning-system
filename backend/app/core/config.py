"""Application configuration loaded from environment variables

Settings are checked at startup. Missing required values raise an
error on first access. That's what we want: fail loud and early
rather than run with the wrong values and find out later.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Environment = Literal["local", "test", "production"]

DEFAULT_CHROME_PROFILE_PATH = Path.home() / ".config" / "learning-system" / "chrome-profile"


class Settings(BaseSettings):
    """Typed configuration, loaded from the process environment or a local .env file.

    Each field has a description next to it. To add a new setting:
        1. Add the field here with a type and a default (or no default if required).
        2. Add the variable to .env.example with a comment.
        3. Read it through get_settings() in the code that needs it.
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

    chrome_profile_path: Path = Field(
        default=DEFAULT_CHROME_PROFILE_PATH,
        description=(
            "Persistent Chrome profile directory used by the Playwright "
            "transport. Holds login cookies and session state for "
            "claude.ai. The directory contains active session credentials "
            "after login; do not commit, share, or sync it."
        ),
    )

    deepseek_api_key: SecretStr = Field(
        description=(
            "API key for the DeepSeek chat completions endpoint. "
            "SecretStr keeps it out of logs and reprs."
        ),
    )

    deepseek_model: str = Field(
        default="deepseek-v4-flash",
        description=(
            "DeepSeek model identifier. Default is the V4 efficiency tier. "
            "Switch to deepseek-v4-pro for harder topics. Legacy aliases "
            "deepseek-chat and deepseek-reasoner retire 2026-07-24."
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
