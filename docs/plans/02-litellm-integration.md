# 02: LiteLLM Integration

## Objective

Create a unified LLM API wrapper using LiteLLM that supports OpenAI, Anthropic, and Google.

## Tasks

- [ ] Create base completion wrapper
- [ ] Add async support
- [ ] Implement fallback configuration
- [ ] Add retry logic configuration
- [ ] Implement streaming support
- [ ] Add cost tracking helpers

## API Design

```python
from progression_labs.llm import complete, acompletion, LLMClient

# Simple usage
response = await complete(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
)

# With fallbacks
response = await complete(
    model="gpt-4o",
    messages=[...],
    fallbacks=["claude-sonnet-4-20250514", "gemini/gemini-2.0-flash"],
)

# Client-based usage
client = LLMClient(
    default_model="gpt-4o",
    fallbacks=["claude-sonnet-4-20250514"],
    max_retries=3,
)
response = await client.complete(messages=[...])
```

## Implementation

```python
# src/progression_labs/llm/completion.py

from typing import Any
import litellm
from litellm import acompletion as litellm_acompletion

async def complete(
    model: str,
    messages: list[dict[str, str]],
    fallbacks: list[str] | None = None,
    max_retries: int = 3,
    timeout: float = 60.0,
    **kwargs: Any,
) -> litellm.ModelResponse:
    """
    Unified async completion across all providers.

    Args:
        model: Model identifier (e.g., "gpt-4o", "claude-sonnet-4-20250514")
        messages: List of message dicts with role and content
        fallbacks: Optional list of fallback models
        max_retries: Number of retries on failure
        timeout: Request timeout in seconds
        **kwargs: Additional args passed to LiteLLM

    Returns:
        ModelResponse with completion and usage info
    """
    return await litellm_acompletion(
        model=model,
        messages=messages,
        fallbacks=fallbacks,
        num_retries=max_retries,
        timeout=timeout,
        **kwargs,
    )


def get_cost(response: litellm.ModelResponse) -> float:
    """Get the cost in USD for a completion response."""
    return litellm.completion_cost(response)
```

## Streaming Support

```python
# src/progression_labs/llm/completion.py

from collections.abc import AsyncIterator

async def stream(
    model: str,
    messages: list[dict[str, str]],
    **kwargs: Any,
) -> AsyncIterator[str]:
    """
    Stream completion tokens.

    Yields:
        Individual tokens as they arrive
    """
    response = await litellm_acompletion(
        model=model,
        messages=messages,
        stream=True,
        **kwargs,
    )
    async for chunk in response:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
```

## Configuration

```python
# src/progression_labs/llm/config.py

from pydantic_settings import BaseSettings

class LLMSettings(BaseSettings):
    """LLM configuration from environment."""

    # API Keys
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None

    # Defaults
    default_model: str = "gpt-4o"
    default_timeout: float = 60.0
    default_max_retries: int = 3

    class Config:
        env_prefix = "LLM_"
```

## Tests

```python
# tests/test_completion.py

import pytest
from progression_labs.llm import complete, get_cost

@pytest.mark.asyncio
async def test_complete_openai():
    response = await complete(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello"}],
    )
    assert response.choices[0].message.content
    assert response.usage.total_tokens > 0

@pytest.mark.asyncio
async def test_get_cost():
    response = await complete(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hi"}],
    )
    cost = get_cost(response)
    assert cost > 0
```

## Acceptance Criteria

- [ ] `complete()` works with OpenAI models
- [ ] `complete()` works with Anthropic models
- [ ] `complete()` works with Google models
- [ ] Fallbacks work when primary model fails
- [ ] Retries work on transient errors
- [ ] Streaming works
- [ ] Cost tracking returns accurate values
- [ ] All functions are async-first
