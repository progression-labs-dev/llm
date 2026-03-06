"""Comprehensive tests for VectorStore component."""

import tempfile
from unittest.mock import patch

import pytest

from progression_labs.llm.rag import SearchResult, VectorStore


class TestVectorStoreInitialization:
    """Tests for VectorStore initialization."""

    def test_creates_in_memory_client_by_default(self):
        """Test VectorStore creates in-memory client when no persist_directory."""
        store = VectorStore(collection_name="test_memory")
        # Verify store is functional (internal client was created successfully)
        assert store.count() == 0

    def test_creates_persistent_client_with_directory(self):
        """Test VectorStore creates persistent client with persist_directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(
                collection_name="test_persist",
                persist_directory=tmpdir,
            )
            # Verify store is functional (internal client was created successfully)
            assert store.count() == 0

    def test_stores_embedding_model(self):
        """Test VectorStore stores embedding model setting."""
        store = VectorStore(
            collection_name="test_model",
            embedding_model="text-embedding-3-large",
        )
        assert store.embedding_model == "text-embedding-3-large"

    def test_stores_collection_name(self):
        """Test VectorStore stores collection name."""
        store = VectorStore(collection_name="my_collection")
        assert store.collection_name == "my_collection"

    @pytest.mark.asyncio
    async def test_uses_cosine_similarity(self):
        """Test VectorStore uses cosine similarity (verified via search behavior)."""
        # Cosine similarity is verified by checking that identical embeddings
        # return high similarity scores (close to 1.0)
        mock_embedding = [1.0, 0.0, 0.0]  # Unit vector

        with patch("progression_labs.llm.rag.vectorstore.embed", return_value=[mock_embedding]):
            store = VectorStore(collection_name="test_cosine")
            await store.add(documents=["test"], ids=["id1"])

        with patch("progression_labs.llm.rag.vectorstore.embed_single", return_value=mock_embedding):
            results = await store.search("test", top_k=1)

        # With cosine similarity, identical vectors should have score close to 1.0
        assert results[0].score > 0.99


class TestVectorStorePersistence:
    """Tests for VectorStore persistence."""

    @pytest.mark.asyncio
    async def test_data_persists_across_instances(self):
        """Test data is persisted and survives new instance creation."""
        mock_embedding = [0.1, 0.2, 0.3]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create store and add data
            with patch("progression_labs.llm.rag.vectorstore.embed", return_value=[mock_embedding]):
                store1 = VectorStore(
                    collection_name="persist_test",
                    persist_directory=tmpdir,
                )
                await store1.add(documents=["test doc"], ids=["doc1"])
                assert store1.count() == 1

            # Create new instance pointing to same directory
            store2 = VectorStore(
                collection_name="persist_test",
                persist_directory=tmpdir,
            )
            assert store2.count() == 1

    @pytest.mark.asyncio
    async def test_different_collections_are_isolated(self):
        """Test different collection names are isolated."""
        mock_embedding = [0.1, 0.2, 0.3]

        with patch("progression_labs.llm.rag.vectorstore.embed", return_value=[mock_embedding]):
            store1 = VectorStore(collection_name="collection_a")
            store2 = VectorStore(collection_name="collection_b")

            await store1.add(documents=["doc1"], ids=["id1"])

            assert store1.count() == 1
            assert store2.count() == 0


class TestVectorStoreAdd:
    """Tests for VectorStore.add() method."""

    @pytest.mark.asyncio
    async def test_add_single_document(self):
        """Test adding a single document."""
        mock_embedding = [0.1, 0.2, 0.3]

        with patch("progression_labs.llm.rag.vectorstore.embed", return_value=[mock_embedding]):
            store = VectorStore(collection_name="test_add_single")
            await store.add(documents=["hello"], ids=["id1"])

        assert store.count() == 1

    @pytest.mark.asyncio
    async def test_add_multiple_documents(self):
        """Test adding multiple documents at once."""
        mock_embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        with patch("progression_labs.llm.rag.vectorstore.embed", return_value=mock_embeddings):
            store = VectorStore(collection_name="test_add_multiple")
            await store.add(
                documents=["doc1", "doc2", "doc3"],
                ids=["id1", "id2", "id3"],
            )

        assert store.count() == 3

    @pytest.mark.asyncio
    async def test_add_with_metadata(self):
        """Test adding documents with metadata."""
        mock_embedding = [0.1, 0.2, 0.3]

        with patch("progression_labs.llm.rag.vectorstore.embed", return_value=[mock_embedding]):
            store = VectorStore(collection_name="test_add_meta")
            await store.add(
                documents=["test"],
                ids=["id1"],
                metadata=[{"source": "test", "page": 1}],
            )

        assert store.count() == 1

    @pytest.mark.asyncio
    async def test_add_duplicate_id_is_ignored(self):
        """Test adding duplicate ID is silently ignored by ChromaDB."""
        mock_embedding = [0.1, 0.2, 0.3]

        with patch("progression_labs.llm.rag.vectorstore.embed", return_value=[mock_embedding]):
            store = VectorStore(collection_name="test_duplicate")
            await store.add(documents=["doc1"], ids=["id1"])

            # ChromaDB silently ignores duplicate IDs
            await store.add(documents=["doc2"], ids=["id1"])

            # Count remains 1 (duplicate was ignored)
            assert store.count() == 1

    @pytest.mark.asyncio
    async def test_add_uses_configured_embedding_model(self):
        """Test add uses the configured embedding model."""
        mock_embedding = [0.1, 0.2, 0.3]

        with patch(
            "progression_labs.llm.rag.vectorstore.embed", return_value=[mock_embedding]
        ) as mock_embed:
            store = VectorStore(
                collection_name="test_embed_model",
                embedding_model="custom-model",
            )
            await store.add(documents=["test"], ids=["id1"])

            mock_embed.assert_called_once_with(["test"], model="custom-model")


class TestVectorStoreSearch:
    """Tests for VectorStore.search() method."""

    @pytest.mark.asyncio
    async def test_search_empty_store_returns_empty_list(self):
        """Test searching empty store returns empty list."""
        with patch("progression_labs.llm.rag.vectorstore.embed_single", return_value=[0.1, 0.2]):
            store = VectorStore(collection_name="test_search_empty")
            results = await store.search("query", top_k=5)

        assert results == []

    @pytest.mark.asyncio
    async def test_search_returns_search_results(self):
        """Test search returns list of SearchResult objects."""
        mock_embedding = [0.1, 0.2, 0.3]

        with patch("progression_labs.llm.rag.vectorstore.embed", return_value=[mock_embedding]):
            store = VectorStore(collection_name="test_search_type")
            await store.add(documents=["test doc"], ids=["id1"])

        with patch("progression_labs.llm.rag.vectorstore.embed_single", return_value=mock_embedding):
            results = await store.search("test", top_k=1)

        assert len(results) == 1
        assert isinstance(results[0], SearchResult)

    @pytest.mark.asyncio
    async def test_search_result_contains_document(self):
        """Test search result contains the original document text."""
        mock_embedding = [0.1, 0.2, 0.3]

        with patch("progression_labs.llm.rag.vectorstore.embed", return_value=[mock_embedding]):
            store = VectorStore(collection_name="test_search_doc")
            await store.add(documents=["Hello world"], ids=["id1"])

        with patch("progression_labs.llm.rag.vectorstore.embed_single", return_value=mock_embedding):
            results = await store.search("hello", top_k=1)

        assert results[0].document == "Hello world"

    @pytest.mark.asyncio
    async def test_search_result_contains_id(self):
        """Test search result contains the document ID."""
        mock_embedding = [0.1, 0.2, 0.3]

        with patch("progression_labs.llm.rag.vectorstore.embed", return_value=[mock_embedding]):
            store = VectorStore(collection_name="test_search_id")
            await store.add(documents=["test"], ids=["my_doc_id"])

        with patch("progression_labs.llm.rag.vectorstore.embed_single", return_value=mock_embedding):
            results = await store.search("test", top_k=1)

        assert results[0].id == "my_doc_id"

    @pytest.mark.asyncio
    async def test_search_result_contains_metadata(self):
        """Test search result contains document metadata."""
        mock_embedding = [0.1, 0.2, 0.3]

        with patch("progression_labs.llm.rag.vectorstore.embed", return_value=[mock_embedding]):
            store = VectorStore(collection_name="test_search_meta")
            await store.add(
                documents=["test"],
                ids=["id1"],
                metadata=[{"source": "file.txt", "page": 5}],
            )

        with patch("progression_labs.llm.rag.vectorstore.embed_single", return_value=mock_embedding):
            results = await store.search("test", top_k=1)

        assert results[0].metadata == {"source": "file.txt", "page": 5}

    @pytest.mark.asyncio
    async def test_search_result_score_is_similarity(self):
        """Test search result score is similarity (0-1) not distance."""
        mock_embedding = [0.1, 0.2, 0.3]

        with patch("progression_labs.llm.rag.vectorstore.embed", return_value=[mock_embedding]):
            store = VectorStore(collection_name="test_search_score")
            await store.add(documents=["test"], ids=["id1"])

        with patch("progression_labs.llm.rag.vectorstore.embed_single", return_value=mock_embedding):
            results = await store.search("test", top_k=1)

        # Score should be high (close to 1) for identical embeddings
        assert 0 <= results[0].score <= 1
        assert results[0].score > 0.9  # Should be very high for same embedding

    @pytest.mark.asyncio
    async def test_search_respects_top_k(self):
        """Test search returns at most top_k results."""
        mock_embeddings = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

        with patch("progression_labs.llm.rag.vectorstore.embed", return_value=mock_embeddings):
            store = VectorStore(collection_name="test_topk")
            await store.add(
                documents=["doc1", "doc2", "doc3"],
                ids=["id1", "id2", "id3"],
            )

        with patch("progression_labs.llm.rag.vectorstore.embed_single", return_value=[0.1, 0.2]):
            results = await store.search("query", top_k=2)

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_search_top_k_larger_than_docs(self):
        """Test search with top_k larger than number of documents."""
        mock_embedding = [0.1, 0.2, 0.3]

        with patch("progression_labs.llm.rag.vectorstore.embed", return_value=[mock_embedding]):
            store = VectorStore(collection_name="test_topk_large")
            await store.add(documents=["only doc"], ids=["id1"])

        with patch("progression_labs.llm.rag.vectorstore.embed_single", return_value=mock_embedding):
            results = await store.search("query", top_k=100)

        assert len(results) == 1  # Only 1 doc exists

    @pytest.mark.asyncio
    async def test_search_with_metadata_filter(self):
        """Test search with metadata where filter."""
        mock_embeddings = [[0.1, 0.2], [0.1, 0.2]]  # Same embeddings

        with patch("progression_labs.llm.rag.vectorstore.embed", return_value=mock_embeddings):
            store = VectorStore(collection_name="test_where")
            await store.add(
                documents=["python doc", "javascript doc"],
                ids=["id1", "id2"],
                metadata=[{"lang": "python"}, {"lang": "javascript"}],
            )

        with patch("progression_labs.llm.rag.vectorstore.embed_single", return_value=[0.1, 0.2]):
            results = await store.search(
                "programming",
                top_k=10,
                where={"lang": "python"},
            )

        assert len(results) == 1
        assert results[0].metadata["lang"] == "python"


class TestVectorStoreDelete:
    """Tests for VectorStore.delete() method."""

    @pytest.mark.asyncio
    async def test_delete_single_document(self):
        """Test deleting a single document."""
        mock_embedding = [0.1, 0.2, 0.3]

        with patch("progression_labs.llm.rag.vectorstore.embed", return_value=[mock_embedding]):
            store = VectorStore(collection_name="test_delete_single")
            await store.add(documents=["doc"], ids=["id1"])
            assert store.count() == 1

            await store.delete(ids=["id1"])
            assert store.count() == 0

    @pytest.mark.asyncio
    async def test_delete_multiple_documents(self):
        """Test deleting multiple documents at once."""
        mock_embeddings = [[0.1], [0.2], [0.3]]

        with patch("progression_labs.llm.rag.vectorstore.embed", return_value=mock_embeddings):
            store = VectorStore(collection_name="test_delete_multi")
            await store.add(
                documents=["a", "b", "c"],
                ids=["id1", "id2", "id3"],
            )
            assert store.count() == 3

            await store.delete(ids=["id1", "id3"])
            assert store.count() == 1

    @pytest.mark.asyncio
    async def test_delete_nonexistent_id_no_error(self):
        """Test deleting non-existent ID doesn't raise error."""
        store = VectorStore(collection_name="test_delete_nonexistent")
        # Should not raise
        await store.delete(ids=["does_not_exist"])
        assert store.count() == 0


class TestVectorStoreClear:
    """Tests for VectorStore.clear() method."""

    @pytest.mark.asyncio
    async def test_clear_removes_all_documents(self):
        """Test clear removes all documents from collection."""
        mock_embeddings = [[0.1], [0.2], [0.3]]

        with patch("progression_labs.llm.rag.vectorstore.embed", return_value=mock_embeddings):
            store = VectorStore(collection_name="test_clear_all")
            await store.add(
                documents=["a", "b", "c"],
                ids=["1", "2", "3"],
            )
            assert store.count() == 3

            store.clear()
            assert store.count() == 0

    @pytest.mark.asyncio
    async def test_clear_allows_reuse(self):
        """Test store can be reused after clear."""
        mock_embedding = [0.1, 0.2]

        with patch("progression_labs.llm.rag.vectorstore.embed", return_value=[mock_embedding]):
            store = VectorStore(collection_name="test_clear_reuse")
            await store.add(documents=["first"], ids=["id1"])
            store.clear()

            await store.add(documents=["second"], ids=["id2"])
            assert store.count() == 1


class TestVectorStoreCount:
    """Tests for VectorStore.count() method."""

    def test_count_empty_store(self):
        """Test count returns 0 for empty store."""
        store = VectorStore(collection_name="test_count_empty")
        assert store.count() == 0

    @pytest.mark.asyncio
    async def test_count_after_add(self):
        """Test count reflects added documents."""
        mock_embeddings = [[0.1], [0.2]]

        with patch("progression_labs.llm.rag.vectorstore.embed", return_value=mock_embeddings):
            store = VectorStore(collection_name="test_count_add")
            await store.add(documents=["a", "b"], ids=["1", "2"])

        assert store.count() == 2

    @pytest.mark.asyncio
    async def test_count_after_delete(self):
        """Test count reflects deleted documents."""
        mock_embeddings = [[0.1], [0.2]]

        with patch("progression_labs.llm.rag.vectorstore.embed", return_value=mock_embeddings):
            store = VectorStore(collection_name="test_count_delete")
            await store.add(documents=["a", "b"], ids=["1", "2"])
            await store.delete(ids=["1"])

        assert store.count() == 1


class TestSearchResultDataclass:
    """Tests for SearchResult dataclass."""

    def test_all_fields_accessible(self):
        """Test all SearchResult fields are accessible."""
        result = SearchResult(
            id="test_id",
            document="test doc",
            score=0.85,
            metadata={"key": "value"},
        )

        assert result.id == "test_id"
        assert result.document == "test doc"
        assert result.score == 0.85
        assert result.metadata == {"key": "value"}

    def test_empty_metadata(self):
        """Test SearchResult with empty metadata."""
        result = SearchResult(
            id="id",
            document="doc",
            score=0.5,
            metadata={},
        )
        assert result.metadata == {}

    def test_equality(self):
        """Test SearchResult equality comparison."""
        result1 = SearchResult(id="id", document="doc", score=0.5, metadata={})
        result2 = SearchResult(id="id", document="doc", score=0.5, metadata={})

        assert result1 == result2
