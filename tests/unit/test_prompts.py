"""Tests for Langfuse prompt management utilities."""

from unittest.mock import MagicMock, patch

import progression_labs.llm.prompts as prompts_module
from progression_labs.llm import get_chat_prompt, get_langfuse, get_prompt, get_prompt_with_vars


class TestGetLangfuse:
    """Tests for get_langfuse function."""

    def test_returns_langfuse_client(self):
        """Test that get_langfuse returns a Langfuse client."""
        with patch("progression_labs.llm.prompts.Langfuse") as mock_langfuse:
            mock_client = MagicMock()
            mock_langfuse.return_value = mock_client

            # Reset the singleton
            prompts_module._langfuse = None

            client = get_langfuse()
            assert client == mock_client
            mock_langfuse.assert_called_once()

    def test_returns_singleton(self):
        """Test that get_langfuse returns the same instance."""
        with patch("progression_labs.llm.prompts.Langfuse") as mock_langfuse:
            mock_client = MagicMock()
            mock_langfuse.return_value = mock_client

            # Reset the singleton
            prompts_module._langfuse = None

            client1 = get_langfuse()
            client2 = get_langfuse()

            assert client1 is client2
            # Should only be called once due to singleton
            mock_langfuse.assert_called_once()


class TestGetPrompt:
    """Tests for get_prompt function."""

    def test_fetches_prompt_by_name(self):
        """Test fetching a prompt by name."""
        mock_client = MagicMock()
        mock_prompt = MagicMock()
        mock_prompt.compile.return_value = "Hello, world!"
        mock_client.get_prompt.return_value = mock_prompt

        # Set mock client
        prompts_module._langfuse = mock_client

        result = get_prompt("greeting")

        mock_client.get_prompt.assert_called_once_with("greeting", version=None, label=None)
        assert result == "Hello, world!"

    def test_fetches_prompt_with_version(self):
        """Test fetching a specific prompt version."""
        mock_client = MagicMock()
        mock_prompt = MagicMock()
        mock_prompt.compile.return_value = "Hello v2!"
        mock_client.get_prompt.return_value = mock_prompt

        prompts_module._langfuse = mock_client

        result = get_prompt("greeting", version=2)

        mock_client.get_prompt.assert_called_once_with("greeting", version=2, label=None)
        assert result == "Hello v2!"

    def test_fetches_prompt_with_label(self):
        """Test fetching a prompt by label."""
        mock_client = MagicMock()
        mock_prompt = MagicMock()
        mock_prompt.compile.return_value = "Staging prompt"
        mock_client.get_prompt.return_value = mock_prompt

        prompts_module._langfuse = mock_client

        result = get_prompt("greeting", label="staging")

        mock_client.get_prompt.assert_called_once_with("greeting", version=None, label="staging")
        assert result == "Staging prompt"


class TestGetPromptWithVars:
    """Tests for get_prompt_with_vars function."""

    def test_compiles_prompt_with_variables(self):
        """Test compiling a prompt with variables."""
        mock_client = MagicMock()
        mock_prompt = MagicMock()
        mock_prompt.compile.return_value = "Hello, Alice!"
        mock_client.get_prompt.return_value = mock_prompt

        prompts_module._langfuse = mock_client

        result = get_prompt_with_vars("greeting", {"name": "Alice"})

        mock_client.get_prompt.assert_called_once_with("greeting", version=None, label=None)
        mock_prompt.compile.assert_called_once_with(name="Alice")
        assert result == "Hello, Alice!"

    def test_compiles_with_multiple_variables(self):
        """Test compiling a prompt with multiple variables."""
        mock_client = MagicMock()
        mock_prompt = MagicMock()
        mock_prompt.compile.return_value = "Hello, Alice! You have 5 messages."
        mock_client.get_prompt.return_value = mock_prompt

        prompts_module._langfuse = mock_client

        result = get_prompt_with_vars("greeting", {"name": "Alice", "count": "5"})

        mock_prompt.compile.assert_called_once_with(name="Alice", count="5")
        assert result == "Hello, Alice! You have 5 messages."


class TestGetChatPrompt:
    """Tests for get_chat_prompt function."""

    def test_fetches_chat_prompt(self):
        """Test fetching a chat prompt."""
        mock_client = MagicMock()
        mock_prompt = MagicMock()
        expected_messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        mock_prompt.compile.return_value = expected_messages
        mock_client.get_prompt.return_value = mock_prompt

        prompts_module._langfuse = mock_client

        result = get_chat_prompt("chat_template")

        mock_client.get_prompt.assert_called_once_with(
            "chat_template", version=None, label=None, type="chat"
        )
        assert result == expected_messages

    def test_fetches_chat_prompt_with_variables(self):
        """Test fetching a chat prompt with variables."""
        mock_client = MagicMock()
        mock_prompt = MagicMock()
        expected_messages = [
            {"role": "system", "content": "You are helping Alice."},
        ]
        mock_prompt.compile.return_value = expected_messages
        mock_client.get_prompt.return_value = mock_prompt

        prompts_module._langfuse = mock_client

        result = get_chat_prompt("chat_template", {"user_name": "Alice"})

        mock_prompt.compile.assert_called_once_with(user_name="Alice")
        assert result == expected_messages
