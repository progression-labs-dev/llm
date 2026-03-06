"""Integration tests for LLM completion functionality.

These tests call real LLM providers. Requires API keys
(loaded via conftest.py GCP Secret Manager fixture).
"""

import pytest

from progression_labs.llm import complete, get_cost, get_usage, stream

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_complete_openai():
    """Test completion with OpenAI model."""
    response = await complete(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
    )

    assert response.choices is not None
    assert len(response.choices) > 0
    assert response.choices[0].message.content is not None
    assert "hello" in response.choices[0].message.content.lower()


@pytest.mark.asyncio
async def test_complete_anthropic():
    """Test completion with Anthropic model."""
    response = await complete(
        model="claude-3-5-haiku-20241022",
        messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
    )

    assert response.choices is not None
    assert len(response.choices) > 0
    assert response.choices[0].message.content is not None
    assert "hello" in response.choices[0].message.content.lower()


@pytest.mark.asyncio
async def test_get_cost():
    """Test cost tracking."""
    response = await complete(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hi"}],
    )

    cost = get_cost(response)
    assert cost > 0
    assert isinstance(cost, float)


@pytest.mark.asyncio
async def test_get_usage():
    """Test token usage tracking."""
    response = await complete(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hi"}],
    )

    usage = get_usage(response)
    assert usage.prompt_tokens > 0
    assert usage.completion_tokens > 0
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens


@pytest.mark.asyncio
async def test_stream():
    """Test streaming completion."""
    tokens = []
    async for token in stream(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Count from 1 to 3."}],
    ):
        tokens.append(token)

    full_response = "".join(tokens)
    assert len(tokens) > 0
    assert "1" in full_response
    assert "2" in full_response
    assert "3" in full_response


@pytest.mark.asyncio
async def test_complete_with_system_message():
    """Test completion with system message."""
    response = await complete(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a pirate. Respond like a pirate."},
            {"role": "user", "content": "Hello!"},
        ],
    )

    content = response.choices[0].message.content.lower()
    # Pirates typically say things like "ahoy", "arr", "matey"
    assert any(word in content for word in ["ahoy", "arr", "matey", "aye", "pirate"])
