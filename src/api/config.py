"""API configuration."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class APISettings(BaseSettings):
    """API settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API
    api_title: str = "RecSys E-commerce API"
    api_version: str = "0.1.0"
    api_description: str = "Recommendation API for e-commerce platform"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True

    # Models
    default_model: str = "popular"
    default_n_items: int = 10
    max_n_items: int = 100


@lru_cache
def get_api_settings() -> APISettings:
    """Get cached API settings."""
    return APISettings()
