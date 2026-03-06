# 03: Instructor Integration

## Objective

Integrate Instructor for structured output with Pydantic validation on all LLM calls.

## Tasks

- [ ] Create Instructor client wrapper
- [ ] Support LiteLLM backend
- [ ] Add retry configuration for validation failures
- [ ] Implement streaming for partial objects
- [ ] Add nested object support

## API Design

```python
from pydantic import BaseModel
from progression_labs.llm import extract, aextract

class User(BaseModel):
    name: str
    age: int
    email: str

# Simple extraction
user = await extract(
    response_model=User,
    model="gpt-4o",
    messages=[{"role": "user", "content": "John is 25, email john@example.com"}],
)
print(user.name)  # "John"
print(user.age)   # 25

# With custom prompt
user = await extract(
    response_model=User,
    model="gpt-4o",
    prompt="Extract user info: John Doe, age 30, john.doe@company.com",
)
```

## Implementation

```python
# src/progression_labs/llm/structured.py

from typing import TypeVar
import instructor
import litellm
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

# Create instructor client with LiteLLM backend
_client = instructor.from_litellm(litellm.acompletion)


async def extract(
    response_model: type[T],
    model: str,
    messages: list[dict[str, str]] | None = None,
    prompt: str | None = None,
    max_retries: int = 3,
    **kwargs,
) -> T:
    """
    Extract structured data from LLM response.

    Args:
        response_model: Pydantic model class defining the output schema
        model: Model identifier (e.g., "gpt-4o")
        messages: Optional message list (mutually exclusive with prompt)
        prompt: Optional simple prompt string
        max_retries: Retries on validation failure
        **kwargs: Additional args passed to LiteLLM

    Returns:
        Instance of response_model with extracted data

    Raises:
        ValidationError: If extraction fails after retries
    """
    if prompt and not messages:
        messages = [{"role": "user", "content": prompt}]

    if not messages:
        raise ValueError("Either messages or prompt must be provided")

    return await _client(
        model=model,
        messages=messages,
        response_model=response_model,
        max_retries=max_retries,
        **kwargs,
    )
```

## Streaming Partial Objects

```python
# src/progression_labs/llm/structured.py

from collections.abc import AsyncIterator
from instructor import Partial

async def extract_stream(
    response_model: type[T],
    model: str,
    messages: list[dict[str, str]],
    **kwargs,
) -> AsyncIterator[Partial[T]]:
    """
    Stream partial extractions as they arrive.

    Yields:
        Partial[T] objects with fields populated as they're extracted
    """
    async for partial in _client(
        model=model,
        messages=messages,
        response_model=Partial[response_model],
        stream=True,
        **kwargs,
    ):
        yield partial
```

## Complex Nested Models

```python
# Example usage with nested models

from pydantic import BaseModel, Field

class Address(BaseModel):
    street: str
    city: str
    country: str

class Company(BaseModel):
    name: str
    industry: str
    headquarters: Address
    employees: list[str] = Field(default_factory=list)

# Works with nested structures
company = await extract(
    response_model=Company,
    model="gpt-4o",
    prompt="""
    Extract company info:
    Acme Corp is a technology company based at 123 Main St,
    San Francisco, USA. Employees include Alice, Bob, and Charlie.
    """,
)
print(company.headquarters.city)  # "San Francisco"
print(company.employees)          # ["Alice", "Bob", "Charlie"]
```

## Validation with Custom Rules

```python
from pydantic import BaseModel, field_validator

class UserWithValidation(BaseModel):
    name: str
    age: int
    email: str

    @field_validator("age")
    @classmethod
    def validate_age(cls, v: int) -> int:
        if v < 0 or v > 150:
            raise ValueError("Age must be between 0 and 150")
        return v

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("Invalid email format")
        return v

# Instructor will retry if validation fails
user = await extract(
    response_model=UserWithValidation,
    model="gpt-4o",
    prompt="John is 25 years old, email: john@example.com",
    max_retries=3,  # Retries if validation fails
)
```

## Tests

```python
# tests/test_structured.py

import pytest
from pydantic import BaseModel
from progression_labs.llm import extract

class SimpleUser(BaseModel):
    name: str
    age: int

@pytest.mark.asyncio
async def test_extract_simple():
    user = await extract(
        response_model=SimpleUser,
        model="gpt-4o-mini",
        prompt="Alice is 30 years old",
    )
    assert user.name == "Alice"
    assert user.age == 30

@pytest.mark.asyncio
async def test_extract_with_validation():
    user = await extract(
        response_model=SimpleUser,
        model="gpt-4o-mini",
        prompt="Bob, aged twenty-five",  # LLM should parse "twenty-five" as 25
    )
    assert user.age == 25
```

## Acceptance Criteria

- [ ] `extract()` returns validated Pydantic models
- [ ] Works with all providers (OpenAI, Anthropic, Google)
- [ ] Retries on validation failure
- [ ] Streaming partial objects works
- [ ] Nested models work
- [ ] Custom validators work
- [ ] Type hints work correctly in IDE
