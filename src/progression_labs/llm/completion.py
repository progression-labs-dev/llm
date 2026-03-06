"""
Unified LLM completion wrapper using LiteLLM.

Provides async-first interface for OpenAI, Anthropic, and Google models
with configurable retry logic and exponential backoff.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import litellm
from litellm import ModelResponse
from litellm import acompletion as litellm_acompletion
from pydantic import BaseModel

from progression_labs.llm.config import LLMSettings, get_settings


class UsageStats(BaseModel):
    """Token usage statistics from a completion response."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class RetryConfig:
    """Configuration for retry behavior with exponential backoff."""

    max_retries: int = 3
    min_wait: float = 1.0  # Minimum wait between retries (seconds)
    max_wait: float = 60.0  # Maximum wait between retries (seconds)
    multiplier: float = 2.0  # Exponential backoff multiplier

    @classmethod
    def from_settings(cls, settings: LLMSettings | None = None) -> "RetryConfig":
        """Create RetryConfig from LLMSettings."""
        s = settings or get_settings()
        return cls(
            max_retries=s.default_max_retries,
            min_wait=s.retry_min_wait,
            max_wait=s.retry_max_wait,
            multiplier=s.retry_multiplier,
        )


async def complete(
    model: str,
    messages: list[dict[str, str]],
    fallbacks: list[str] | None = None,
    max_retries: int | None = None,
    timeout: float | None = None,
    retry_config: RetryConfig | None = None,
    settings: LLMSettings | None = None,
    project: str | None = None,
    feature: str | None = None,
    user_id: str | None = None,
    request_id: str | None = None,
    **kwargs: Any,
) -> ModelResponse:
    """
    Unified async completion across all providers.

    Supports configurable retry logic with exponential backoff.

    Args:
        model: Model identifier (e.g., "gpt-4o", "claude-sonnet-4-20250514",
               "gemini/gemini-2.0-flash")
        messages: List of message dicts with role and content
        fallbacks: Optional list of fallback models if primary fails
        max_retries: Number of retries on transient failures (overrides retry_config)
        timeout: Request timeout in seconds (uses settings default if not provided)
        retry_config: Optional RetryConfig for fine-grained retry control
        settings: Optional custom LLMSettings instance (uses global if not provided)
        project: Project identifier for observability tracking
        feature: Feature or use case name for observability tracking
        user_id: User who triggered the call
        request_id: Correlation ID linking to app logs
        **kwargs: Additional args passed to LiteLLM

    Returns:
        ModelResponse with completion and usage info

    Example:
        >>> response = await complete(
        ...     model="gpt-4o",
        ...     messages=[{"role": "user", "content": "Hello!"}],
        ...     project="my-app",
        ...     feature="chat",
        ... )
        >>> print(response.choices[0].message.content)
    """
    s = settings or get_settings()
    rc = retry_config or RetryConfig.from_settings(s)

    # Allow max_retries to override retry_config for backwards compatibility
    effective_retries = max_retries if max_retries is not None else rc.max_retries
    effective_timeout = timeout if timeout is not None else s.default_timeout

    # Build LLM call metadata (LLM Services standard)
    call_metadata: dict[str, str] = {}
    if project:
        call_metadata["project"] = project
    if feature:
        call_metadata["feature"] = feature
    if user_id:
        call_metadata["user_id"] = user_id
    if request_id:
        call_metadata["request_id"] = request_id
    if call_metadata:
        kwargs["metadata"] = {**kwargs.get("metadata", {}), **call_metadata}

    return await litellm_acompletion(
        model=model,
        messages=messages,
        fallbacks=fallbacks,
        num_retries=effective_retries,
        timeout=effective_timeout,
        **kwargs,
    )


async def stream(
    model: str,
    messages: list[dict[str, str]],
    timeout: float | None = None,
    settings: LLMSettings | None = None,
    project: str | None = None,
    feature: str | None = None,
    user_id: str | None = None,
    request_id: str | None = None,
    **kwargs: Any,
) -> AsyncIterator[str]:
    """
    Stream completion tokens.

    Args:
        model: Model identifier
        messages: List of message dicts
        timeout: Request timeout in seconds (uses settings default if not provided)
        settings: Optional custom LLMSettings instance (uses global if not provided)
        project: Project identifier for observability tracking
        feature: Feature or use case name for observability tracking
        user_id: User who triggered the call
        request_id: Correlation ID linking to app logs
        **kwargs: Additional args passed to LiteLLM

    Yields:
        Individual tokens as they arrive

    Example:
        >>> async for token in stream(
        ...     model="gpt-4o",
        ...     messages=[{"role": "user", "content": "Tell me a story"}],
        ...     project="my-app",
        ...     feature="chat",
        ... ):
        ...     print(token, end="", flush=True)
    """
    s = settings or get_settings()
    effective_timeout = timeout if timeout is not None else s.default_timeout

    # Build LLM call metadata (LLM Services standard)
    call_metadata: dict[str, str] = {}
    if project:
        call_metadata["project"] = project
    if feature:
        call_metadata["feature"] = feature
    if user_id:
        call_metadata["user_id"] = user_id
    if request_id:
        call_metadata["request_id"] = request_id
    if call_metadata:
        kwargs["metadata"] = {**kwargs.get("metadata", {}), **call_metadata}

    response = await litellm_acompletion(
        model=model,
        messages=messages,
        stream=True,
        timeout=effective_timeout,
        **kwargs,
    )
    async for chunk in response:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def get_cost(response: ModelResponse) -> float:
    """
    Get the cost in USD for a completion response.

    Args:
        response: The ModelResponse from a completion call

    Returns:
        Cost in USD

    Example:
        >>> response = await complete(model="gpt-4o", messages=[...])
        >>> cost = get_cost(response)
        >>> print(f"Cost: ${cost:.6f}")
    """
    return litellm.completion_cost(response)


def get_usage(response: ModelResponse) -> UsageStats:
    """
    Get token usage from a completion response.

    Args:
        response: The ModelResponse from a completion call

    Returns:
        UsageStats with prompt_tokens, completion_tokens, total_tokens

    Raises:
        ValueError: If usage information is not available in the response

    Example:
        >>> response = await complete(model="gpt-4o", messages=[...])
        >>> usage = get_usage(response)
        >>> print(f"Total tokens: {usage.total_tokens}")
    """
    # Access via model_dump() — litellm's ModelResponse uses Pydantic extra='allow'
    # which ty cannot resolve for direct attribute access.
    data = response.model_dump()
    usage = data.get("usage")
    if usage is None:
        raise ValueError("Usage information not available in response")
    return UsageStats(
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        total_tokens=usage["total_tokens"],
    )
