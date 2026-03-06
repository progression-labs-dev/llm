"""
Vector store abstraction over ChromaDB.

Provides a simple interface for storing and searching document embeddings,
designed for easy migration to Qdrant or other vector stores later.

The internal storage implementation is fully encapsulated - no direct access
to the underlying ChromaDB client or collection is exposed.
"""

from dataclasses import dataclass
from typing import cast

import chromadb
from chromadb.api.types import Embedding, Metadata

from .embeddings import embed, embed_single


@dataclass
class SearchResult:
    """A single search result from the vector store."""

    id: str
    document: str
    score: float
    metadata: dict


class VectorStore:
    """
    Simple vector store abstraction over ChromaDB.

    Designed for easy migration to Qdrant or other vector stores later.
    The internal storage implementation is fully encapsulated.

    Example:
        >>> store = VectorStore(collection_name="docs")
        >>> await store.add(
        ...     documents=["Python is great", "JavaScript is popular"],
        ...     ids=["doc1", "doc2"],
        ... )
        >>> results = await store.search("programming language", top_k=1)
        >>> results[0].document
        'Python is great'
    """

    __slots__ = (
        "embedding_model",
        "collection_name",
        "_VectorStore__client",
        "_VectorStore__collection",
    )

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
        self.collection_name = collection_name

        if persist_directory:
            self.__client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.__client = chromadb.Client()

        self.__collection = self.__client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    async def add(
        self,
        documents: list[str],
        ids: list[str],
        metadata: list[dict[str, str | int | float | bool]] | None = None,
    ) -> None:
        """
        Add documents to the store.

        Args:
            documents: List of document texts
            ids: Unique IDs for each document
            metadata: Optional metadata for each document

        Example:
            >>> await store.add(
            ...     documents=["Doc 1 content", "Doc 2 content"],
            ...     ids=["doc1", "doc2"],
            ...     metadata=[{"source": "file1"}, {"source": "file2"}],
            ... )
        """
        raw_embeddings = await embed(documents, model=self.embedding_model)
        # Cast to chromadb's Embedding type — list[list[float]] is compatible
        # with list[Sequence[float]], but list invariance prevents direct assignment.
        embeddings = cast(list[Embedding], raw_embeddings)

        # Cast to chromadb's Metadata type — our dict values are compatible but
        # chromadb's Metadata union includes SparseVector which widens the type.
        metadatas = cast(list[Metadata], metadata) if metadata is not None else None

        self.__collection.add(
            documents=documents,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas,
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
            List of SearchResult objects sorted by relevance

        Example:
            >>> results = await store.search("find relevant docs", top_k=5)
            >>> for r in results:
            ...     print(f"{r.id}: {r.score:.2f}")
        """
        query_embedding = await embed_single(query, model=self.embedding_model)

        results = self.__collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results["ids"] and results["ids"][0]:
            distances = results["distances"]
            metadatas = results["metadatas"]
            documents = results["documents"]
            for i in range(len(results["ids"][0])):
                score = (1 - distances[0][i]) if distances else 0.0
                meta_item = metadatas[0][i] if metadatas else None
                search_results.append(
                    SearchResult(
                        id=results["ids"][0][i],
                        document=documents[0][i] if documents else "",
                        score=score,
                        metadata=dict(meta_item) if meta_item is not None else {},
                    )
                )

        return search_results

    async def delete(self, ids: list[str]) -> None:
        """
        Delete documents by ID.

        Args:
            ids: List of document IDs to delete
        """
        self.__collection.delete(ids=ids)

    def count(self) -> int:
        """Get number of documents in the store."""
        return self.__collection.count()

    def clear(self) -> None:
        """Clear all documents from the collection."""
        self.__client.delete_collection(name=self.collection_name)
        self.__collection = self.__client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
