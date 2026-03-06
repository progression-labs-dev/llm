# 05: ChromaDB RAG

## Objective

Implement simple RAG functionality using ChromaDB for vector storage and LiteLLM for embeddings.

## Tasks

- [ ] Create embeddings wrapper using LiteLLM
- [ ] Implement ChromaDB vector store abstraction
- [ ] Add document chunking utilities
- [ ] Create simple retrieve-and-generate function
- [ ] Design for future Qdrant migration

## API Design

```python
from progression_labs.llm.rag import VectorStore, embed, retrieve_and_generate

# Create vector store
store = VectorStore(collection_name="docs")

# Add documents
await store.add(
    documents=["Doc 1 content", "Doc 2 content"],
    ids=["doc1", "doc2"],
    metadata=[{"source": "file1"}, {"source": "file2"}],
)

# Search
results = await store.search(query="find relevant docs", top_k=5)

# Simple RAG
answer = await retrieve_and_generate(
    query="What is X?",
    store=store,
    model="gpt-4o",
)
```

## Implementation

### Embeddings

```python
# src/progression_labs/llm/rag/embeddings.py

import litellm

async def embed(
    texts: list[str],
    model: str = "text-embedding-3-small",
) -> list[list[float]]:
    """
    Generate embeddings for texts using LiteLLM.

    Args:
        texts: List of texts to embed
        model: Embedding model (default: OpenAI text-embedding-3-small)

    Returns:
        List of embedding vectors
    """
    response = await litellm.aembedding(
        model=model,
        input=texts,
    )
    return [item["embedding"] for item in response.data]


async def embed_single(
    text: str,
    model: str = "text-embedding-3-small",
) -> list[float]:
    """Generate embedding for a single text."""
    embeddings = await embed([text], model=model)
    return embeddings[0]
```

### Vector Store

```python
# src/progression_labs/llm/rag/vectorstore.py

from dataclasses import dataclass
import chromadb
from chromadb.config import Settings

from .embeddings import embed, embed_single


@dataclass
class SearchResult:
    """A single search result."""
    id: str
    document: str
    score: float
    metadata: dict


class VectorStore:
    """
    Simple vector store abstraction over ChromaDB.

    Designed for easy migration to Qdrant later.
    """

    def __init__(
        self,
        collection_name: str,
        persist_directory: str | None = None,
        embedding_model: str = "text-embedding-3-small",
    ):
        """
        Initialize vector store.

        Args:
            collection_name: Name of the collection
            persist_directory: Optional directory for persistence
            embedding_model: Model for generating embeddings
        """
        self.embedding_model = embedding_model

        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def add(
        self,
        documents: list[str],
        ids: list[str],
        metadata: list[dict] | None = None,
    ) -> None:
        """
        Add documents to the store.

        Args:
            documents: List of document texts
            ids: Unique IDs for each document
            metadata: Optional metadata for each document
        """
        embeddings = await embed(documents, model=self.embedding_model)

        self._collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadata,
        )

    async def search(
        self,
        query: str,
        top_k: int = 5,
        where: dict | None = None,
    ) -> list[SearchResult]:
        """
        Search for similar documents.

        Args:
            query: Search query text
            top_k: Number of results to return
            where: Optional metadata filter

        Returns:
            List of SearchResult objects
        """
        query_embedding = await embed_single(query, model=self.embedding_model)

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        for i in range(len(results["ids"][0])):
            search_results.append(SearchResult(
                id=results["ids"][0][i],
                document=results["documents"][0][i],
                score=1 - results["distances"][0][i],  # Convert distance to similarity
                metadata=results["metadatas"][0][i] if results["metadatas"] else {},
            ))

        return search_results

    async def delete(self, ids: list[str]) -> None:
        """Delete documents by ID."""
        self._collection.delete(ids=ids)

    def count(self) -> int:
        """Get number of documents in the store."""
        return self._collection.count()
```

### Retrieve and Generate

```python
# src/progression_labs/llm/rag/generate.py

from ..completion import complete
from .vectorstore import VectorStore


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
        system_prompt: Optional custom system prompt

    Returns:
        Generated answer
    """
    # Retrieve
    results = await store.search(query=query, top_k=top_k)

    # Build context
    context = "\n\n".join([
        f"[{i+1}] {r.document}"
        for i, r in enumerate(results)
    ])

    # Generate
    default_system = """You are a helpful assistant. Answer the user's question based on the provided context. If the context doesn't contain relevant information, say so.

Context:
{context}"""

    system = (system_prompt or default_system).format(context=context)

    response = await complete(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ],
    )

    return response.choices[0].message.content
```

## Tests

```python
# tests/test_rag.py

import pytest
from progression_labs.llm.rag import VectorStore, embed, retrieve_and_generate

@pytest.mark.asyncio
async def test_embed():
    embeddings = await embed(["hello world"])
    assert len(embeddings) == 1
    assert len(embeddings[0]) > 0  # Has dimensions

@pytest.mark.asyncio
async def test_vector_store():
    store = VectorStore(collection_name="test")

    await store.add(
        documents=["Python is a programming language", "JavaScript runs in browsers"],
        ids=["doc1", "doc2"],
    )

    results = await store.search("programming language", top_k=1)
    assert len(results) == 1
    assert "Python" in results[0].document

@pytest.mark.asyncio
async def test_retrieve_and_generate():
    store = VectorStore(collection_name="test_rag")
    await store.add(
        documents=["The capital of France is Paris."],
        ids=["fact1"],
    )

    answer = await retrieve_and_generate(
        query="What is the capital of France?",
        store=store,
        model="gpt-4o-mini",
    )

    assert "Paris" in answer
```

## Migration Path to Qdrant

The `VectorStore` interface is designed for easy migration:

```python
# Future: src/progression_labs/llm/rag/qdrant_store.py

from qdrant_client import QdrantClient

class QdrantVectorStore:
    """Drop-in replacement using Qdrant."""

    async def add(self, documents, ids, metadata=None): ...
    async def search(self, query, top_k=5, where=None): ...
    async def delete(self, ids): ...
    def count(self): ...
```

## Acceptance Criteria

- [ ] `embed()` generates embeddings via LiteLLM
- [ ] `VectorStore` adds and searches documents
- [ ] Metadata filtering works
- [ ] `retrieve_and_generate()` produces answers
- [ ] Persistence works with `persist_directory`
- [ ] Interface is ready for Qdrant migration
