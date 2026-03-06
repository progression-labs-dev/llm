"""Unit tests for structured output extraction functionality."""

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from progression_labs.llm import extract, extract_stream


class SimpleUser(BaseModel):
    """Simple user model for testing."""

    name: str
    age: int


class TestExtractUnit:
    """Unit tests for extract() with mocked instructor client."""

    @pytest.mark.asyncio
    async def test_extract_with_prompt(self):
        """Prompt is converted to messages."""
        fake_user = SimpleUser(name="Alice", age=30)
        with patch(
            "progression_labs.llm.structured._client.chat.completions.create",
            new_callable=AsyncMock,
            return_value=fake_user,
        ) as mock_create:
            result = await extract(
                response_model=SimpleUser,
                model="gpt-4o",
                prompt="Alice is 30",
            )

        assert result.name == "Alice"
        assert result.age == 30
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["messages"] == [{"role": "user", "content": "Alice is 30"}]

    @pytest.mark.asyncio
    async def test_extract_with_messages(self):
        """Messages are passed through directly."""
        fake_user = SimpleUser(name="Bob", age=25)
        msgs = [
            {"role": "system", "content": "Extract user info."},
            {"role": "user", "content": "Bob is 25."},
        ]
        with patch(
            "progression_labs.llm.structured._client.chat.completions.create",
            new_callable=AsyncMock,
            return_value=fake_user,
        ) as mock_create:
            result = await extract(
                response_model=SimpleUser,
                model="gpt-4o",
                messages=msgs,
            )

        assert result.name == "Bob"
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["messages"] == msgs

    @pytest.mark.asyncio
    async def test_extract_no_input_raises(self):
        """Raises ValueError when neither prompt nor messages provided."""
        with pytest.raises(ValueError, match="Either messages or prompt must be provided"):
            await extract(
                response_model=SimpleUser,
                model="gpt-4o",
            )

    @pytest.mark.asyncio
    async def test_extract_passes_kwargs(self):
        """Extra kwargs are forwarded to instructor."""
        fake_user = SimpleUser(name="X", age=1)
        with patch(
            "progression_labs.llm.structured._client.chat.completions.create",
            new_callable=AsyncMock,
            return_value=fake_user,
        ) as mock_create:
            await extract(
                response_model=SimpleUser,
                model="gpt-4o",
                prompt="X is 1",
                temperature=0.5,
            )

        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_retries"] == 3  # default


@pytest.mark.asyncio
async def test_extract_no_input_raises():
    """Test that extract raises error when no input provided."""
    with pytest.raises(ValueError, match="Either messages or prompt must be provided"):
        await extract(
            response_model=SimpleUser,
            model="gpt-4o-mini",
        )


@pytest.mark.asyncio
async def test_extract_stream_no_input_raises():
    """Test that extract_stream raises error when no input provided."""
    with pytest.raises(ValueError, match="Either messages or prompt must be provided"):
        async for _ in extract_stream(
            response_model=SimpleUser,
            model="gpt-4o-mini",
        ):
            pass
