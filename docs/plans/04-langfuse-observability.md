# 04: Langfuse Observability

## Objective

Integrate Langfuse for LLM-specific observability including traces, prompt versioning, and cost tracking.

## Tasks

- [ ] Set up Langfuse callback with LiteLLM
- [ ] Add trace context helpers
- [ ] Implement prompt management utilities
- [ ] Add custom metadata support
- [ ] Create cost tracking helpers

## API Design

```python
from progression_labs.llm import init_observability, trace

# Initialize at app startup
init_observability()

# Automatic tracing via LiteLLM callbacks
response = await complete(model="gpt-4o", messages=[...])
# ^ Automatically traced to Langfuse

# Manual trace context
@trace(name="process_document")
async def process_document(doc: str) -> Summary:
    # All LLM calls inside are grouped under this trace
    summary = await extract(response_model=Summary, ...)
    return summary

# With metadata
@trace(name="user_chat", metadata={"feature": "support"})
async def handle_chat(user_id: str, message: str) -> str:
    ...
```

## Implementation

```python
# src/progression_labs/llm/observability.py

import os
import litellm
from functools import wraps
from typing import Callable, ParamSpec, TypeVar
from langfuse.decorators import observe, langfuse_context

P = ParamSpec("P")
R = TypeVar("R")


def init_observability() -> None:
    """
    Initialize Langfuse observability.

    Requires environment variables:
        LANGFUSE_PUBLIC_KEY
        LANGFUSE_SECRET_KEY
        LANGFUSE_HOST (optional, defaults to cloud)
    """
    # Validate required env vars
    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        raise ValueError("LANGFUSE_PUBLIC_KEY environment variable required")
    if not os.getenv("LANGFUSE_SECRET_KEY"):
        raise ValueError("LANGFUSE_SECRET_KEY environment variable required")

    # Set up LiteLLM callbacks
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]


def trace(
    name: str | None = None,
    metadata: dict | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator to create a trace context for LLM operations.

    Args:
        name: Trace name (defaults to function name)
        metadata: Custom metadata dict
        user_id: Optional user identifier
        session_id: Optional session identifier
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        @observe(name=name)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Add metadata to current trace
            if metadata:
                langfuse_context.update_current_observation(
                    metadata=metadata
                )
            if user_id:
                langfuse_context.update_current_trace(user_id=user_id)
            if session_id:
                langfuse_context.update_current_trace(session_id=session_id)

            return await func(*args, **kwargs)

        return wrapper
    return decorator


def add_trace_metadata(metadata: dict) -> None:
    """Add metadata to the current trace context."""
    langfuse_context.update_current_observation(metadata=metadata)


def set_trace_user(user_id: str) -> None:
    """Set the user ID for the current trace."""
    langfuse_context.update_current_trace(user_id=user_id)


def set_trace_session(session_id: str) -> None:
    """Set the session ID for the current trace."""
    langfuse_context.update_current_trace(session_id=session_id)
```

## Prompt Management

```python
# src/progression_labs/llm/prompts.py

from langfuse import Langfuse

_langfuse: Langfuse | None = None


def get_langfuse() -> Langfuse:
    """Get or create Langfuse client."""
    global _langfuse
    if _langfuse is None:
        _langfuse = Langfuse()
    return _langfuse


def get_prompt(name: str, version: int | None = None) -> str:
    """
    Fetch a prompt from Langfuse by name.

    Args:
        name: Prompt name in Langfuse
        version: Optional specific version (defaults to latest)

    Returns:
        Prompt template string
    """
    client = get_langfuse()
    prompt = client.get_prompt(name, version=version)
    return prompt.compile()


def get_prompt_with_vars(
    name: str,
    variables: dict[str, str],
    version: int | None = None,
) -> str:
    """
    Fetch and compile a prompt with variables.

    Args:
        name: Prompt name in Langfuse
        variables: Dict of variable substitutions
        version: Optional specific version

    Returns:
        Compiled prompt string with variables substituted
    """
    client = get_langfuse()
    prompt = client.get_prompt(name, version=version)
    return prompt.compile(**variables)
```

## Usage Example

```python
from progression_labs.llm import (
    init_observability,
    trace,
    complete,
    extract,
    get_prompt,
)

# Initialize at startup
init_observability()

@trace(name="customer_support", metadata={"team": "support"})
async def handle_support_ticket(ticket_id: str, message: str) -> str:
    # Fetch prompt from Langfuse (versioned)
    system_prompt = get_prompt("support_agent_v2")

    # This call is automatically traced
    response = await complete(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
    )

    return response.choices[0].message.content
```

## Environment Variables

```bash
# Required
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...

# Optional (defaults to Langfuse cloud)
LANGFUSE_HOST=https://cloud.langfuse.com
```

## Tests

```python
# tests/test_observability.py

import pytest
from unittest.mock import patch
from progression_labs.llm import init_observability, trace

def test_init_requires_env_vars():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="LANGFUSE_PUBLIC_KEY"):
            init_observability()

@pytest.mark.asyncio
async def test_trace_decorator():
    @trace(name="test_trace")
    async def my_function() -> str:
        return "result"

    result = await my_function()
    assert result == "result"
```

## Acceptance Criteria

- [ ] `init_observability()` configures LiteLLM callbacks
- [ ] All LLM calls appear in Langfuse
- [ ] `@trace` decorator groups operations
- [ ] Custom metadata appears in traces
- [ ] User/session IDs can be set
- [ ] Prompt fetching from Langfuse works
- [ ] Costs are tracked automatically
