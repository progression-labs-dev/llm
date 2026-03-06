"""
Langfuse observability integration for LLM operations.

Provides automatic tracing, metadata tracking, and cost monitoring.
"""

import os
from collections.abc import Awaitable, Callable
from functools import wraps
from typing import overload

import litellm
from langfuse.decorators import langfuse_context, observe


def init_observability() -> None:
    """
    Initialize Langfuse observability for all LLM calls.

    This configures LiteLLM to send all completion events to Langfuse
    for automatic tracing and cost tracking.

    Requires environment variables:
        LANGFUSE_PUBLIC_KEY: Your Langfuse public key
        LANGFUSE_SECRET_KEY: Your Langfuse secret key
        LANGFUSE_HOST: Optional, defaults to Langfuse cloud

    Raises:
        ValueError: If required environment variables are not set

    Example:
        >>> from progression_labs.llm import init_observability
        >>> init_observability()  # Call once at app startup
    """
    if not os.getenv("LANGFUSE_PUBLIC_KEY"):
        raise ValueError("LANGFUSE_PUBLIC_KEY environment variable required")
    if not os.getenv("LANGFUSE_SECRET_KEY"):
        raise ValueError("LANGFUSE_SECRET_KEY environment variable required")

    # Configure LiteLLM to use Langfuse callbacks
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]


@overload
def trace[**P, R](
    func: Callable[P, Awaitable[R]],
) -> Callable[P, Awaitable[R]]: ...


@overload
def trace[**P, R](
    func: None = None,
    *,
    name: str | None = None,
    metadata: dict | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]: ...


def trace[**P, R](
    func: Callable[P, Awaitable[R]] | None = None,
    *,
    name: str | None = None,
    metadata: dict | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> Callable[P, Awaitable[R]] | Callable[[Callable[P, Awaitable[R]]], Callable[P, Awaitable[R]]]:
    """
    Decorator to create a trace context for LLM operations.

    Groups all LLM calls within the decorated function under a single trace
    in Langfuse. Supports both @trace and @trace(...) syntax.

    Args:
        func: The function to wrap (when used without parentheses)
        name: Trace name (defaults to function name)
        metadata: Custom metadata dict to attach to the trace
        user_id: Optional user identifier for the trace
        session_id: Optional session identifier for the trace

    Returns:
        Decorated async function with tracing enabled

    Example:
        >>> @trace
        ... async def simple_operation():
        ...     return await complete(model="gpt-4o", messages=[...])

        >>> @trace(name="process_doc", metadata={"type": "pdf"})
        ... async def process_document(doc: str) -> Summary:
        ...     return await extract(response_model=Summary, ...)
    """

    def decorator(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
        # Get function name for the trace (Callable may not have __name__, but our decorated fns do)
        fn_name = getattr(fn, "__name__", "unknown")

        @wraps(fn)
        @observe(name=name or fn_name)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            # Add metadata/user/session to current trace using v2 API
            if metadata:
                langfuse_context.update_current_observation(metadata=metadata)
            if user_id:
                langfuse_context.update_current_trace(user_id=user_id)
            if session_id:
                langfuse_context.update_current_trace(session_id=session_id)

            return await fn(*args, **kwargs)

        return wrapper

    # Handle @trace without parentheses
    if func is not None:
        return decorator(func)

    return decorator


def add_trace_metadata(metadata: dict) -> None:
    """
    Add metadata to the current trace context.

    Call this within a @trace decorated function to attach additional
    metadata discovered during execution.

    Args:
        metadata: Dict of metadata to add

    Example:
        >>> @trace(name="process")
        ... async def process(doc_id: str):
        ...     doc = await fetch_document(doc_id)
        ...     add_trace_metadata({"doc_size": len(doc)})
        ...     return await summarize(doc)
    """
    langfuse_context.update_current_observation(metadata=metadata)


def set_trace_user(user_id: str) -> None:
    """
    Set the user ID for the current trace.

    Args:
        user_id: User identifier to associate with the trace

    Example:
        >>> @trace
        ... async def handle_request(user_id: str, message: str):
        ...     set_trace_user(user_id)
        ...     return await complete(...)
    """
    langfuse_context.update_current_trace(user_id=user_id)


def set_trace_session(session_id: str) -> None:
    """
    Set the session ID for the current trace.

    Args:
        session_id: Session identifier to associate with the trace

    Example:
        >>> @trace
        ... async def handle_chat(session_id: str, message: str):
        ...     set_trace_session(session_id)
        ...     return await complete(...)
    """
    langfuse_context.update_current_trace(session_id=session_id)


def flush_traces() -> None:
    """
    Flush all pending traces to Langfuse.

    Call this before application shutdown to ensure all traces are sent.
    In most cases this is handled automatically, but it's useful for
    short-lived scripts or serverless functions.

    Example:
        >>> try:
        ...     await main()
        ... finally:
        ...     flush_traces()
    """
    langfuse_context.flush()
