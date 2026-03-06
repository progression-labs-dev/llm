# Structured Output Research

## Recommendation: Instructor

**Confidence: High**

Instructor is the industry standard for structured LLM output with Pydantic validation.

## Overview

Instructor is the most popular Python library for extracting structured data from LLMs. Built on Pydantic, it provides type-safe data extraction with automatic validation, retries, and streaming.

- **Website**: https://python.useinstructor.com/
- **GitHub**: https://github.com/567-labs/instructor
- **PyPI**: https://pypi.org/project/instructor/
- **Stats**: 3M+ monthly downloads, 11k+ stars, 100+ contributors

## Key Features

| Feature | Description |
|---------|-------------|
| **Pydantic Models** | Define exact output schema as Pydantic models |
| **Automatic Retries** | Built-in retry when validation fails |
| **Data Validation** | Leverage Pydantic's validation engine |
| **Streaming** | Real-time processing of partial responses |
| **Multi-Provider** | OpenAI, Anthropic, Google, + 15 more |
| **Type Safety** | Full IDE support with proper type inference |

## Multi-Language Support

Available in Python, TypeScript, Go, Ruby, Elixir, and Rust.

## Usage Example

```python
import instructor
from pydantic import BaseModel
from openai import OpenAI

class User(BaseModel):
    name: str
    age: int

client = instructor.from_openai(OpenAI())

user = client.chat.completions.create(
    model="gpt-4o",
    response_model=User,
    messages=[
        {"role": "user", "content": "Extract: John is 25 years old"}
    ]
)

print(user.name)  # "John"
print(user.age)   # 25
```

## With LiteLLM

```python
import instructor
import litellm

client = instructor.from_litellm(litellm.completion)

user = client(
    model="claude-sonnet-4-20250514",
    response_model=User,
    messages=[
        {"role": "user", "content": "Extract: John is 25 years old"}
    ]
)
```

## Validation with Retries

```python
from pydantic import BaseModel, field_validator

class User(BaseModel):
    name: str
    age: int

    @field_validator("age")
    @classmethod
    def validate_age(cls, v):
        if v < 0 or v > 150:
            raise ValueError("Age must be between 0 and 150")
        return v

# Instructor will automatically retry if validation fails
user = client.chat.completions.create(
    model="gpt-4o",
    response_model=User,
    max_retries=3,
    messages=[...]
)
```

## Streaming Support

```python
from instructor import Partial

for partial_user in client.chat.completions.create_partial(
    model="gpt-4o",
    response_model=User,
    messages=[...],
    stream=True
):
    print(partial_user)  # Partial[User] with available fields
```

## Nested Objects

```python
class Address(BaseModel):
    street: str
    city: str
    country: str

class Company(BaseModel):
    name: str
    address: Address
    employees: list[str]

# Instructor handles complex nested structures automatically
```

## Alternatives Considered

| Tool | Why Not |
|------|---------|
| PydanticAI | Better for agents, Instructor better for extraction |
| Raw JSON mode | No validation, no retries, manual schema |
| Guardrails AI | More focused on safety, less on extraction |
| Outlines | Lower-level, requires more setup |

## Instructor vs PydanticAI

- **Instructor**: Best for schema-first extraction, fast and lightweight
- **PydanticAI**: Best for agents with tools, built-in observability

For our use case (critical structured output on all calls), Instructor is the right choice.

## Integration Notes

- Works with LiteLLM via `instructor.from_litellm()`
- Pydantic models can be reused for API schemas
- Validation errors include the LLM's attempted output for debugging

## Sources

- [Instructor Documentation](https://python.useinstructor.com/)
- [GitHub - instructor](https://github.com/567-labs/instructor)
- [Pydantic LLM Guide](https://pydantic.dev/articles/llm-intro)
- [PydanticAI vs Instructor](https://medium.com/@mahadevan.varadhan/pydanticai-vs-instructor-structured-llm-ai-outputs-with-python-tools-c7b7b202eb23)
