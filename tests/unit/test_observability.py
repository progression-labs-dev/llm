"""Tests for Langfuse observability integration."""

import os
from unittest.mock import patch

import pytest

from progression_labs.llm import (
    add_trace_metadata,
    flush_traces,
    init_observability,
    set_trace_session,
    set_trace_user,
    trace,
)


class TestInitObservability:
    """Tests for init_observability function."""

    def test_requires_public_key(self):
        """Test that init_observability requires LANGFUSE_PUBLIC_KEY."""
        with (
            patch.dict(os.environ, {}, clear=True),
            pytest.raises(ValueError, match="LANGFUSE_PUBLIC_KEY"),
        ):
            init_observability()

    def test_requires_secret_key(self):
        """Test that init_observability requires LANGFUSE_SECRET_KEY."""
        with (
            patch.dict(os.environ, {"LANGFUSE_PUBLIC_KEY": "pk-test"}, clear=True),
            pytest.raises(ValueError, match="LANGFUSE_SECRET_KEY"),
        ):
            init_observability()

    def test_configures_litellm_callbacks(self):
        """Test that init_observability configures LiteLLM callbacks."""
        with patch.dict(
            os.environ,
            {
                "LANGFUSE_PUBLIC_KEY": "pk-test",
                "LANGFUSE_SECRET_KEY": "sk-test",
            },
            clear=True,
        ):
            import litellm

            init_observability()
            assert "langfuse" in litellm.success_callback
            assert "langfuse" in litellm.failure_callback


class TestTraceDecorator:
    """Tests for the @trace decorator."""

    @pytest.mark.asyncio
    async def test_trace_without_args(self):
        """Test @trace decorator without arguments."""

        @trace
        async def simple_function() -> str:
            return "result"

        result = await simple_function()
        assert result == "result"

    @pytest.mark.asyncio
    async def test_trace_with_name(self):
        """Test @trace decorator with custom name."""

        @trace(name="custom_trace")
        async def named_function() -> str:
            return "named result"

        result = await named_function()
        assert result == "named result"

    @pytest.mark.asyncio
    async def test_trace_with_metadata(self):
        """Test @trace decorator with metadata."""

        @trace(metadata={"key": "value"})
        async def metadata_function() -> str:
            return "metadata result"

        result = await metadata_function()
        assert result == "metadata result"

    @pytest.mark.asyncio
    async def test_trace_with_user_id(self):
        """Test @trace decorator with user_id."""

        @trace(user_id="user-123")
        async def user_function() -> str:
            return "user result"

        result = await user_function()
        assert result == "user result"

    @pytest.mark.asyncio
    async def test_trace_with_session_id(self):
        """Test @trace decorator with session_id."""

        @trace(session_id="session-456")
        async def session_function() -> str:
            return "session result"

        result = await session_function()
        assert result == "session result"

    @pytest.mark.asyncio
    async def test_trace_with_all_options(self):
        """Test @trace decorator with all options."""

        @trace(
            name="full_trace",
            metadata={"feature": "test"},
            user_id="user-123",
            session_id="session-456",
        )
        async def full_function() -> str:
            return "full result"

        result = await full_function()
        assert result == "full result"

    @pytest.mark.asyncio
    async def test_trace_preserves_function_name(self):
        """Test that @trace preserves the original function name."""

        @trace
        async def my_named_function() -> str:
            return "result"

        assert my_named_function.__name__ == "my_named_function"

    @pytest.mark.asyncio
    async def test_trace_preserves_docstring(self):
        """Test that @trace preserves the original docstring."""

        @trace
        async def documented_function() -> str:
            """This is the docstring."""
            return "result"

        assert documented_function.__doc__ == "This is the docstring."

    @pytest.mark.asyncio
    async def test_trace_with_args_and_kwargs(self):
        """Test @trace decorator with function arguments."""

        @trace(name="args_test")
        async def function_with_args(a: int, b: str, c: bool = True) -> dict:
            return {"a": a, "b": b, "c": c}

        result = await function_with_args(1, "hello", c=False)
        assert result == {"a": 1, "b": "hello", "c": False}


class TestTraceHelpers:
    """Tests for trace helper functions."""

    def test_add_trace_metadata(self):
        """Test add_trace_metadata calls langfuse_context.update_current_observation."""
        with patch("progression_labs.llm.observability.langfuse_context") as mock_context:
            add_trace_metadata({"key": "value"})
            mock_context.update_current_observation.assert_called_once_with(
                metadata={"key": "value"}
            )

    def test_set_trace_user(self):
        """Test set_trace_user calls langfuse_context.update_current_trace."""
        with patch("progression_labs.llm.observability.langfuse_context") as mock_context:
            set_trace_user("user-123")
            mock_context.update_current_trace.assert_called_once_with(user_id="user-123")

    def test_set_trace_session(self):
        """Test set_trace_session calls langfuse_context.update_current_trace."""
        with patch("progression_labs.llm.observability.langfuse_context") as mock_context:
            set_trace_session("session-456")
            mock_context.update_current_trace.assert_called_once_with(session_id="session-456")

    def test_flush_traces(self):
        """Test flush_traces calls langfuse_context.flush."""
        with patch("progression_labs.llm.observability.langfuse_context") as mock_context:
            flush_traces()
            mock_context.flush.assert_called_once()
