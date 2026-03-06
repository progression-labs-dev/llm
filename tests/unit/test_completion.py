"""Unit tests for LLM completion functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from progression_labs.llm import complete, get_cost, stream
from progression_labs.llm.completion import RetryConfig, UsageStats
from progression_labs.llm.completion import get_usage as _get_usage
from progression_labs.llm.config import LLMSettings


class TestRetryConfig:
    """Tests for RetryConfig dataclass."""

    def test_defaults(self):
        rc = RetryConfig()
        assert rc.max_retries == 3
        assert rc.min_wait == 1.0
        assert rc.max_wait == 60.0
        assert rc.multiplier == 2.0

    def test_from_settings(self):
        settings = LLMSettings(
            default_max_retries=5,
            retry_min_wait=2.0,
            retry_max_wait=120.0,
            retry_multiplier=3.0,
        )
        rc = RetryConfig.from_settings(settings)
        assert rc.max_retries == 5
        assert rc.min_wait == 2.0
        assert rc.max_wait == 120.0
        assert rc.multiplier == 3.0

    def test_from_settings_uses_global_when_none(self):
        rc = RetryConfig.from_settings(None)
        assert rc.max_retries == 3  # default


class TestComplete:
    """Unit tests for complete() with mocked litellm."""

    @pytest.mark.asyncio
    async def test_complete_calls_litellm(self):
        mock_response = MagicMock()
        with patch(
            "progression_labs.llm.completion.litellm_acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_acomp:
            result = await complete(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hi"}],
            )

        assert result is mock_response
        mock_acomp.assert_called_once()
        call_kwargs = mock_acomp.call_args[1]
        assert call_kwargs["model"] == "gpt-4o"
        assert call_kwargs["num_retries"] == 3  # default

    @pytest.mark.asyncio
    async def test_complete_metadata_building(self):
        mock_response = MagicMock()
        with patch(
            "progression_labs.llm.completion.litellm_acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_acomp:
            await complete(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hi"}],
                project="my-proj",
                feature="chat",
                user_id="u1",
                request_id="r1",
            )

        call_kwargs = mock_acomp.call_args[1]
        assert call_kwargs["metadata"]["project"] == "my-proj"
        assert call_kwargs["metadata"]["feature"] == "chat"
        assert call_kwargs["metadata"]["user_id"] == "u1"
        assert call_kwargs["metadata"]["request_id"] == "r1"

    @pytest.mark.asyncio
    async def test_complete_max_retries_override(self):
        mock_response = MagicMock()
        with patch(
            "progression_labs.llm.completion.litellm_acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_acomp:
            await complete(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hi"}],
                max_retries=10,
            )

        call_kwargs = mock_acomp.call_args[1]
        assert call_kwargs["num_retries"] == 10

    @pytest.mark.asyncio
    async def test_complete_custom_timeout(self):
        mock_response = MagicMock()
        with patch(
            "progression_labs.llm.completion.litellm_acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_acomp:
            await complete(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hi"}],
                timeout=30.0,
            )

        call_kwargs = mock_acomp.call_args[1]
        assert call_kwargs["timeout"] == 30.0

    @pytest.mark.asyncio
    async def test_complete_no_metadata_when_empty(self):
        mock_response = MagicMock()
        with patch(
            "progression_labs.llm.completion.litellm_acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ) as mock_acomp:
            await complete(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hi"}],
            )

        call_kwargs = mock_acomp.call_args[1]
        assert "metadata" not in call_kwargs


class TestStream:
    """Unit tests for stream() with mocked litellm."""

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello"

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = " world"

        chunk3 = MagicMock()
        chunk3.choices = []  # empty choices, should be skipped

        async def mock_chunks():
            for c in [chunk1, chunk2, chunk3]:
                yield c

        with patch(
            "progression_labs.llm.completion.litellm_acompletion",
            new_callable=AsyncMock,
            return_value=mock_chunks(),
        ):
            tokens = [t async for t in stream(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hi"}],
            )]

        assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_stream_metadata_building(self):
        # Build a real async generator that yields nothing
        chunk = MagicMock()
        chunk.choices = []

        async def mock_chunks():
            for _ in [chunk]:
                yield _

        with patch(
            "progression_labs.llm.completion.litellm_acompletion",
            new_callable=AsyncMock,
            return_value=mock_chunks(),
        ) as mock_acomp:
            async for _ in stream(
                model="gpt-4o",
                messages=[{"role": "user", "content": "Hi"}],
                project="p",
                feature="f",
            ):
                pass

        call_kwargs = mock_acomp.call_args[1]
        assert call_kwargs["metadata"]["project"] == "p"
        assert call_kwargs["metadata"]["feature"] == "f"
        assert call_kwargs["stream"] is True


class TestGetCost:
    """Unit tests for get_cost()."""

    def test_get_cost_delegates_to_litellm(self):
        mock_response = MagicMock()
        with patch("progression_labs.llm.completion.litellm.completion_cost", return_value=0.005):
            cost = get_cost(mock_response)
        assert cost == 0.005


class TestGetUsage:
    """Unit tests for get_usage()."""

    def test_get_usage_returns_stats(self):
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            }
        }
        usage = _get_usage(mock_response)
        assert isinstance(usage, UsageStats)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30

    def test_get_usage_raises_when_no_usage(self):
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {}
        with pytest.raises(ValueError, match="Usage information not available"):
            _get_usage(mock_response)
