"""
Embedding utilities for RAG using LiteLLM.

Provides functions to generate embeddings for texts using various providers.
Includes an optional LRU cache to avoid redundant API calls for repeated texts.
"""

import asyncio
from collections import OrderedDict

import litellm
from pydantic import BaseModel


class CacheStats(BaseModel):
    """Statistics for the embedding cache."""

    size: int
    max_size: int


# Simple async-compatible LRU cache for embeddings
_embedding_cache: OrderedDict[tuple[str, str], list[float]] = OrderedDict()
_cache_lock = asyncio.Lock()
_cache_max_size = 1000


async def _get_cached_embedding(text: str, model: str) -> list[float] | None:
    """Get embedding from cache if available."""
    key = (text, model)
    async with _cache_lock:
        if key in _embedding_cache:
            # Move to end (most recently used)
            _embedding_cache.move_to_end(key)
            return _embedding_cache[key]
    return None


async def _set_cached_embedding(text: str, model: str, embedding: list[float]) -> None:
    """Store embedding in cache."""
    key = (text, model)
    async with _cache_lock:
        if key in _embedding_cache:
            _embedding_cache.move_to_end(key)
        else:
            _embedding_cache[key] = embedding
            # Evict oldest if over capacity
            while len(_embedding_cache) > _cache_max_size:
                _embedding_cache.popitem(last=False)


def configure_embedding_cache(max_size: int = 1000) -> None:
    """
    Configure the embedding cache size.

    Args:
        max_size: Maximum number of embeddings to cache (default: 1000)

    Example:
        >>> configure_embedding_cache(max_size=5000)
    """
    global _cache_max_size
    _cache_max_size = max_size


def clear_embedding_cache() -> None:
    """
    Clear all cached embeddings.

    Useful for testing or when memory is constrained.

    Example:
        >>> clear_embedding_cache()
    """
    _embedding_cache.clear()


def get_embedding_cache_stats() -> CacheStats:
    """
    Get cache statistics.

    Returns:
        CacheStats with size and max_size

    Example:
        >>> stats = get_embedding_cache_stats()
        >>> print(f"Cache: {stats.size}/{stats.max_size}")
    """
    return CacheStats(size=len(_embedding_cache), max_size=_cache_max_size)


async def embed(
    texts: list[str],
    model: str = "text-embedding-3-small",
    use_cache: bool = True,
) -> list[list[float]]:
    """
    Generate embeddings for texts using LiteLLM.

    Uses an LRU cache to avoid redundant API calls for repeated texts.

    Args:
        texts: List of texts to embed
        model: Embedding model (default: OpenAI text-embedding-3-small)
        use_cache: Whether to use embedding cache (default: True)

    Returns:
        List of embedding vectors

    Example:
        >>> embeddings = await embed(["Hello world", "Goodbye world"])
        >>> len(embeddings)
        2
    """
    if not use_cache:
        response = await litellm.aembedding(model=model, input=texts)
        return [item["embedding"] for item in response.data]

    # Check cache for each text
    results: list[list[float] | None] = [None] * len(texts)
    texts_to_embed: list[tuple[int, str]] = []

    for i, text in enumerate(texts):
        cached = await _get_cached_embedding(text, model)
        if cached is not None:
            results[i] = cached
        else:
            texts_to_embed.append((i, text))

    # Fetch uncached embeddings
    if texts_to_embed:
        uncached_texts = [t for _, t in texts_to_embed]
        response = await litellm.aembedding(model=model, input=uncached_texts)

        for (original_idx, text), item in zip(texts_to_embed, response.data, strict=True):
            embedding = item["embedding"]
            results[original_idx] = embedding
            await _set_cached_embedding(text, model, embedding)

    return results  # type: ignore[return-value]


async def embed_single(
    text: str,
    model: str = "text-embedding-3-small",
    use_cache: bool = True,
) -> list[float]:
    """
    Generate embedding for a single text.

    Uses an LRU cache to avoid redundant API calls for repeated texts.

    Args:
        text: Text to embed
        model: Embedding model (default: OpenAI text-embedding-3-small)
        use_cache: Whether to use embedding cache (default: True)

    Returns:
        Embedding vector

    Example:
        >>> embedding = await embed_single("Hello world")
        >>> len(embedding) > 0
        True
    """
    embeddings = await embed([text], model=model, use_cache=use_cache)
    return embeddings[0]
