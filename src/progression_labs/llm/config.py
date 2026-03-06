"""
Configuration for the LLM package.

Uses pydantic-settings to load from environment variables.
Supports dependency injection for testing flexibility.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """LLM configuration loaded from environment variables."""

    model_config = SettingsConfigDict(
        extra="ignore",
    )

    # API Keys (loaded from environment)
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None

    # Defaults
    default_model: str = "gpt-4o"
    default_timeout: float = 60.0
    default_max_retries: int = 3

    # Retry configuration
    retry_min_wait: float = 1.0  # Minimum wait between retries (seconds)
    retry_max_wait: float = 60.0  # Maximum wait between retries (seconds)
    retry_multiplier: float = 2.0  # Exponential backoff multiplier

    # Langfuse (for observability)
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # OTLP Configuration (for metrics export)
    otlp_endpoint: str = "http://localhost:4317"
    is_otlp_insecure: bool = True
    service_name: str = "progression-labs-llm"
    service_environment: str = "development"


# Global settings instance
_settings: LLMSettings | None = None


def get_settings() -> LLMSettings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = LLMSettings()
    return _settings


def configure(**kwargs: str | float | int | None) -> None:
    """
    Configure LLM settings programmatically.

    Args:
        **kwargs: Settings to override

    Example:
        >>> configure(default_model="claude-sonnet-4-20250514", default_timeout=30.0)
    """
    global _settings
    current = get_settings().model_dump()
    current.update({k: v for k, v in kwargs.items() if v is not None})
    _settings = LLMSettings(**current)


def reset_settings() -> None:
    """
    Reset the global settings instance.

    Useful for testing to ensure a clean state between tests.
    The next call to get_settings() will create a fresh instance.

    Example:
        >>> reset_settings()
        >>> settings = get_settings()  # Fresh instance
    """
    global _settings
    _settings = None


def set_settings(settings: LLMSettings) -> None:
    """
    Set a custom settings instance.

    Enables dependency injection for testing with custom configurations.

    Args:
        settings: Custom LLMSettings instance to use

    Example:
        >>> custom = LLMSettings(default_model="gpt-4o-mini")
        >>> set_settings(custom)
        >>> get_settings().default_model
        'gpt-4o-mini'
    """
    global _settings
    _settings = settings
