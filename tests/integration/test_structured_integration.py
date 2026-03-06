"""Integration tests for structured output extraction functionality.

These tests call real LLM providers. Requires API keys
(loaded via conftest.py GCP Secret Manager fixture).
"""

import pytest
from pydantic import BaseModel, Field, field_validator

from progression_labs.llm import extract, extract_stream

pytestmark = pytest.mark.integration


class SimpleUser(BaseModel):
    """Simple user model for testing."""

    name: str
    age: int


class Address(BaseModel):
    """Address model for nested testing."""

    street: str
    city: str
    country: str


class Company(BaseModel):
    """Company with nested address."""

    name: str
    industry: str
    headquarters: Address


class UserWithValidation(BaseModel):
    """User model with custom validation."""

    name: str
    age: int = Field(ge=0, le=150)
    email: str

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("Invalid email format")
        return v


@pytest.mark.asyncio
async def test_extract_simple():
    """Test basic extraction with simple model."""
    user = await extract(
        response_model=SimpleUser,
        model="gpt-4o-mini",
        prompt="Alice is 30 years old",
    )

    assert user.name == "Alice"
    assert user.age == 30


@pytest.mark.asyncio
async def test_extract_with_messages():
    """Test extraction with message list instead of prompt."""
    user = await extract(
        response_model=SimpleUser,
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract user information from the text."},
            {"role": "user", "content": "Bob is 25 years old."},
        ],
    )

    assert user.name == "Bob"
    assert user.age == 25


@pytest.mark.asyncio
async def test_extract_nested_model():
    """Test extraction with nested Pydantic models."""
    company = await extract(
        response_model=Company,
        model="gpt-4o-mini",
        prompt="""
        Extract company info:
        Acme Corp is a technology company based at 123 Main St,
        San Francisco, USA.
        """,
    )

    assert company.name == "Acme Corp"
    assert company.industry.lower() == "technology"
    assert company.headquarters.city == "San Francisco"
    assert company.headquarters.country in ["USA", "United States", "US"]


@pytest.mark.asyncio
async def test_extract_with_validation():
    """Test extraction with custom validators."""
    user = await extract(
        response_model=UserWithValidation,
        model="gpt-4o-mini",
        prompt="John is 25 years old, email: john@example.com",
        max_retries=3,
    )

    assert user.name == "John"
    assert user.age == 25
    assert "@" in user.email


@pytest.mark.asyncio
async def test_extract_anthropic():
    """Test extraction with Anthropic model."""
    user = await extract(
        response_model=SimpleUser,
        model="claude-3-5-haiku-20241022",
        prompt="Charlie is 40 years old",
    )

    assert user.name == "Charlie"
    assert user.age == 40


@pytest.mark.asyncio
async def test_extract_stream():
    """Test streaming partial extraction."""
    partials = []
    async for partial in extract_stream(
        response_model=SimpleUser,
        model="gpt-4o-mini",
        prompt="David is 35 years old",
    ):
        partials.append(partial)

    # Should have received multiple partial updates
    assert len(partials) > 0

    # Final partial should have complete data
    final = partials[-1]
    assert final.name == "David"
    assert final.age == 35
