"""
Langfuse prompt management utilities.

Provides versioned prompt fetching and compilation from Langfuse.
"""

from typing import cast

from langfuse import Langfuse

_langfuse: Langfuse | None = None


def get_langfuse() -> Langfuse:
    """
    Get or create a Langfuse client instance.

    Returns a singleton Langfuse client configured from environment variables.

    Returns:
        Langfuse client instance

    Example:
        >>> client = get_langfuse()
        >>> prompt = client.get_prompt("my_prompt")
    """
    global _langfuse
    if _langfuse is None:
        _langfuse = Langfuse()
    return _langfuse


def get_prompt(name: str, *, version: int | None = None, label: str | None = None) -> str:
    """
    Fetch a prompt template from Langfuse by name.

    Retrieves a versioned prompt from Langfuse's prompt management system.
    Use this for prompts that need to be versioned and managed separately
    from code.

    Args:
        name: Prompt name as defined in Langfuse
        version: Optional specific version number (defaults to latest production)
        label: Optional label to fetch (e.g., "production", "staging")

    Returns:
        Prompt template string

    Example:
        >>> system_prompt = get_prompt("support_agent")
        >>> # Or fetch a specific version
        >>> system_prompt = get_prompt("support_agent", version=2)
        >>> # Or fetch by label
        >>> system_prompt = get_prompt("support_agent", label="staging")
    """
    client = get_langfuse()
    prompt = client.get_prompt(name, version=version, label=label)
    return prompt.compile()


def get_prompt_with_vars(
    name: str,
    variables: dict[str, str],
    *,
    version: int | None = None,
    label: str | None = None,
) -> str:
    """
    Fetch and compile a prompt with variable substitution.

    Retrieves a prompt from Langfuse and substitutes variables using
    Langfuse's templating syntax ({{variable_name}}).

    Args:
        name: Prompt name as defined in Langfuse
        variables: Dict mapping variable names to values
        version: Optional specific version number
        label: Optional label to fetch

    Returns:
        Compiled prompt string with variables substituted

    Example:
        >>> # If prompt template is: "Hello {{name}}, you have {{count}} messages"
        >>> prompt = get_prompt_with_vars(
        ...     "greeting",
        ...     {"name": "Alice", "count": "5"},
        ... )
        >>> # Returns: "Hello Alice, you have 5 messages"
    """
    client = get_langfuse()
    prompt = client.get_prompt(name, version=version, label=label)
    return prompt.compile(**variables)


def get_chat_prompt(
    name: str,
    variables: dict[str, str] | None = None,
    *,
    version: int | None = None,
    label: str | None = None,
) -> list[dict[str, str]]:
    """
    Fetch a chat prompt (multi-message) from Langfuse.

    Retrieves a chat-style prompt with multiple messages (system, user, assistant)
    from Langfuse and optionally compiles variables.

    Args:
        name: Prompt name as defined in Langfuse
        variables: Optional dict of variable substitutions
        version: Optional specific version number
        label: Optional label to fetch

    Returns:
        List of message dicts with 'role' and 'content' keys,
        ready to pass to complete() or extract()

    Example:
        >>> messages = get_chat_prompt("support_chat", {"user_name": "Alice"})
        >>> response = await complete(model="gpt-4o", messages=messages)
    """
    client = get_langfuse()
    prompt = client.get_prompt(name, version=version, label=label, type="chat")

    if variables:
        return cast(list[dict[str, str]], prompt.compile(**variables))
    return cast(list[dict[str, str]], prompt.compile())
