"""
Structured output extraction using Instructor with LiteLLM backend.

Provides Pydantic-validated responses from LLM calls with automatic retries.
"""

from collections.abc import AsyncIterator
from typing import Any, cast

import instructor
from litellm import acompletion
from pydantic import BaseModel

# Create instructor client with LiteLLM backend
_client = instructor.from_litellm(acompletion)


async def extract[T: BaseModel](
    response_model: type[T],
    model: str,
    messages: list[dict[str, str]] | None = None,
    prompt: str | None = None,
    max_retries: int = 3,
    **kwargs: Any,
) -> T:
    """
    Extract structured data from LLM response.

    Uses Instructor to ensure the LLM output matches the Pydantic model schema.
    Automatically retries on validation failures.

    Args:
        response_model: Pydantic model class defining the output schema
        model: Model identifier (e.g., "gpt-4o", "claude-sonnet-4-20250514")
        messages: Optional message list (mutually exclusive with prompt)
        prompt: Optional simple prompt string (converted to user message)
        max_retries: Number of retries on validation failure
        **kwargs: Additional args passed to LiteLLM

    Returns:
        Instance of response_model with extracted data

    Raises:
        ValueError: If neither messages nor prompt is provided
        ValidationError: If extraction fails after all retries

    Example:
        >>> from pydantic import BaseModel
        >>> class User(BaseModel):
        ...     name: str
        ...     age: int
        >>> user = await extract(
        ...     response_model=User,
        ...     model="gpt-4o",
        ...     prompt="John is 25 years old",
        ... )
        >>> print(user.name)  # "John"
        >>> print(user.age)   # 25
    """
    if prompt and not messages:
        messages = [{"role": "user", "content": prompt}]

    if not messages:
        raise ValueError("Either messages or prompt must be provided")

    result = await _client.chat.completions.create(
        model=model,
        messages=messages,
        response_model=response_model,
        max_retries=max_retries,
        **kwargs,
    )
    return cast(T, result)


async def extract_stream[T: BaseModel](
    response_model: type[T],
    model: str,
    messages: list[dict[str, str]] | None = None,
    prompt: str | None = None,
    **kwargs: Any,
) -> AsyncIterator[T]:
    """
    Stream partial extractions as they arrive.

    Yields partial objects with fields populated incrementally as the LLM
    generates them. Useful for showing progressive UI updates.

    Args:
        response_model: Pydantic model class defining the output schema
        model: Model identifier
        messages: Optional message list (mutually exclusive with prompt)
        prompt: Optional simple prompt string
        **kwargs: Additional args passed to LiteLLM

    Yields:
        Partial instances of response_model with fields populated as available

    Raises:
        ValueError: If neither messages nor prompt is provided

    Example:
        >>> class Story(BaseModel):
        ...     title: str
        ...     content: str
        >>> async for partial in extract_stream(
        ...     response_model=Story,
        ...     model="gpt-4o",
        ...     prompt="Write a short story about a robot",
        ... ):
        ...     print(f"Title: {partial.title}")
        ...     print(f"Content so far: {partial.content}")
    """
    if prompt and not messages:
        messages = [{"role": "user", "content": prompt}]

    if not messages:
        raise ValueError("Either messages or prompt must be provided")

    stream = _client.chat.completions.create_partial(
        model=model,
        messages=messages,
        response_model=response_model,
        **kwargs,
    )

    async for partial in stream:
        yield partial
