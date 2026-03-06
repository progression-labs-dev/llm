"""
RAG (Retrieval-Augmented Generation) module.

Provides embeddings, vector storage, and retrieve-and-generate functionality.

Example:
    >>> from progression_labs.llm.rag import VectorStore, embed, retrieve_and_generate
    >>>
    >>> # Create vector store
    >>> store = VectorStore(collection_name="docs")
    >>>
    >>> # Add documents
    >>> await store.add(
    ...     documents=["Doc 1 content", "Doc 2 content"],
    ...     ids=["doc1", "doc2"],
    ... )
    >>>
    >>> # Search
    >>> results = await store.search("find relevant docs", top_k=5)
    >>>
    >>> # Simple RAG
    >>> answer = await retrieve_and_generate(
    ...     query="What is X?",
    ...     store=store,
    ...     model="gpt-4o",
    ... )
"""

from .embeddings import (
    CacheStats,
    clear_embedding_cache,
    configure_embedding_cache,
    embed,
    embed_single,
    get_embedding_cache_stats,
)
from .generate import retrieve_and_generate
from .vectorstore import SearchResult, VectorStore

__all__ = [
    "embed",
    "embed_single",
    "clear_embedding_cache",
    "configure_embedding_cache",
    "get_embedding_cache_stats",
    "CacheStats",
    "retrieve_and_generate",
    "SearchResult",
    "VectorStore",
]
