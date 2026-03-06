"""
RAG generation utilities.

Provides retrieve-and-generate functionality combining vector search with LLM generation.
"""

from ..completion import complete
from .vectorstore import SearchResult, VectorStore

DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. Answer the user's question based on the \
provided context. If the context doesn't contain relevant information, say so.

Context:
{context}"""


async def retrieve_and_generate(
    query: str,
    store: VectorStore,
    model: str = "gpt-4o",
    top_k: int = 5,
    system_prompt: str | None = None,
) -> str:
    """
    Simple RAG: retrieve relevant docs and generate answer.

    Args:
        query: User query
        store: Vector store to search
        model: LLM model for generation
        top_k: Number of documents to retrieve
        system_prompt: Optional custom system prompt (use {context} placeholder)

    Returns:
        Generated answer

    Example:
        >>> store = VectorStore(collection_name="knowledge")
        >>> await store.add(
        ...     documents=["The capital of France is Paris."],
        ...     ids=["fact1"],
        ... )
        >>> answer = await retrieve_and_generate(
        ...     query="What is the capital of France?",
        ...     store=store,
        ...     model="gpt-4o-mini",
        ... )
        >>> "Paris" in answer
        True
    """
    # Retrieve
    results = await store.search(query=query, top_k=top_k)

    # Build context
    context = _format_context(results)

    # Generate
    system = (system_prompt or DEFAULT_SYSTEM_PROMPT).format(context=context)

    response = await complete(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ],
    )

    # Access via model_dump() — litellm's Choices | StreamingChoices union
    # means ty cannot prove .message always exists on the choice variant.
    data = response.model_dump()
    choices = data.get("choices", [])
    return choices[0]["message"]["content"] or "" if choices else ""


def _format_context(results: list[SearchResult]) -> str:
    """Format search results into context string."""
    if not results:
        return "No relevant documents found."

    return "\n\n".join([f"[{i + 1}] {r.document}" for i, r in enumerate(results)])
