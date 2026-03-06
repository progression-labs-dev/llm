"""Tests for RAG functionality."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from progression_labs.llm.rag import (
    SearchResult,
    VectorStore,
    embed,
    embed_single,
    retrieve_and_generate,
)


class TestEmbed:
    """Tests for embedding functions."""

    @pytest.mark.asyncio
    async def test_embed_returns_embeddings(self):
        """Test embed returns list of embedding vectors."""
        mock_response = MagicMock()
        mock_response.data = [
            {"embedding": [0.1, 0.2, 0.3]},
            {"embedding": [0.4, 0.5, 0.6]},
        ]

        with patch(
            "progression_labs.llm.rag.embeddings.litellm.aembedding",
            return_value=mock_response,
        ):
            embeddings = await embed(["hello", "world"])

        assert len(embeddings) == 2
        assert embeddings[0] == [0.1, 0.2, 0.3]
        assert embeddings[1] == [0.4, 0.5, 0.6]

    @pytest.mark.asyncio
    async def test_embed_single_returns_vector(self):
        """Test embed_single returns single embedding vector."""
        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1, 0.2, 0.3]}]

        with patch(
            "progression_labs.llm.rag.embeddings.litellm.aembedding",
            return_value=mock_response,
        ):
            embedding = await embed_single("hello")

        assert embedding == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_embed_uses_specified_model(self):
        """Test embed passes model parameter to litellm."""
        mock_response = MagicMock()
        mock_response.data = [{"embedding": [0.1]}]

        with patch(
            "progression_labs.llm.rag.embeddings.litellm.aembedding", return_value=mock_response
        ) as mock_aembedding:
            await embed(["hello"], model="text-embedding-3-large")

        mock_aembedding.assert_called_once_with(
            model="text-embedding-3-large",
            input=["hello"],
        )


class TestVectorStore:
    """Tests for VectorStore class."""

    @pytest.mark.asyncio
    async def test_add_documents(self):
        """Test adding documents to vector store."""
        mock_embedding = [0.1, 0.2, 0.3]

        with patch(
            "progression_labs.llm.rag.vectorstore.embed",
            return_value=[mock_embedding, mock_embedding],
        ):
            store = VectorStore(collection_name="test_add")
            await store.add(
                documents=["doc1", "doc2"],
                ids=["id1", "id2"],
            )

        assert store.count() == 2

    @pytest.mark.asyncio
    async def test_add_documents_with_metadata(self):
        """Test adding documents with metadata."""
        mock_embedding = [0.1, 0.2, 0.3]

        with patch(
            "progression_labs.llm.rag.vectorstore.embed",
            return_value=[mock_embedding],
        ):
            store = VectorStore(collection_name="test_metadata")
            await store.add(
                documents=["doc1"],
                ids=["id1"],
                metadata=[{"source": "test"}],
            )

        assert store.count() == 1

    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        """Test searching returns SearchResult objects."""
        mock_embedding = [0.1, 0.2, 0.3]

        with patch(
            "progression_labs.llm.rag.vectorstore.embed",
            return_value=[mock_embedding, mock_embedding],
        ):
            store = VectorStore(collection_name="test_search")
            await store.add(
                documents=["Python programming", "JavaScript coding"],
                ids=["doc1", "doc2"],
            )

        with patch(
            "progression_labs.llm.rag.vectorstore.embed_single",
            return_value=mock_embedding,
        ):
            results = await store.search("programming", top_k=2)

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)
        assert all(hasattr(r, "id") for r in results)
        assert all(hasattr(r, "document") for r in results)
        assert all(hasattr(r, "score") for r in results)

    @pytest.mark.asyncio
    async def test_delete_documents(self):
        """Test deleting documents from store."""
        mock_embedding = [0.1, 0.2, 0.3]

        with patch(
            "progression_labs.llm.rag.vectorstore.embed",
            return_value=[mock_embedding],
        ):
            store = VectorStore(collection_name="test_delete")
            await store.add(documents=["doc1"], ids=["id1"])
            assert store.count() == 1

            await store.delete(ids=["id1"])
            assert store.count() == 0

    def test_count_empty_store(self):
        """Test count on empty store returns 0."""
        store = VectorStore(collection_name="test_empty")
        assert store.count() == 0

    @pytest.mark.asyncio
    async def test_clear_removes_all_documents(self):
        """Test clear removes all documents."""
        mock_embedding = [0.1, 0.2, 0.3]

        with patch(
            "progression_labs.llm.rag.vectorstore.embed",
            return_value=[mock_embedding, mock_embedding],
        ):
            store = VectorStore(collection_name="test_clear")
            await store.add(documents=["doc1", "doc2"], ids=["id1", "id2"])
            assert store.count() == 2

            store.clear()
            assert store.count() == 0


class TestRetrieveAndGenerate:
    """Tests for retrieve_and_generate function."""

    @pytest.mark.asyncio
    async def test_retrieve_and_generate_basic(self):
        """Test basic retrieve and generate flow."""
        mock_store = MagicMock(spec=VectorStore)
        mock_store.search = AsyncMock(
            return_value=[
                SearchResult(
                    id="doc1",
                    document="The capital of France is Paris.",
                    score=0.9,
                    metadata={},
                )
            ]
        )

        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "choices": [{"message": {"content": "The capital of France is Paris."}}],
        }

        with patch("progression_labs.llm.rag.generate.complete", return_value=mock_response):
            answer = await retrieve_and_generate(
                query="What is the capital of France?",
                store=mock_store,
                model="gpt-4o-mini",
            )

        assert "Paris" in answer
        mock_store.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_and_generate_custom_prompt(self):
        """Test retrieve and generate with custom system prompt."""
        mock_store = MagicMock(spec=VectorStore)
        mock_store.search = AsyncMock(return_value=[])

        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "choices": [{"message": {"content": "Custom response"}}],
        }

        with patch(
            "progression_labs.llm.rag.generate.complete", return_value=mock_response
        ) as mock_complete:
            await retrieve_and_generate(
                query="test",
                store=mock_store,
                system_prompt="Custom prompt with {context}",
            )

        # Verify custom prompt was used
        call_args = mock_complete.call_args
        system_message = call_args.kwargs["messages"][0]
        assert "Custom prompt with" in system_message["content"]

    @pytest.mark.asyncio
    async def test_retrieve_and_generate_uses_top_k(self):
        """Test retrieve and generate respects top_k parameter."""
        mock_store = MagicMock(spec=VectorStore)
        mock_store.search = AsyncMock(return_value=[])

        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "choices": [{"message": {"content": "response"}}],
        }

        with patch("progression_labs.llm.rag.generate.complete", return_value=mock_response):
            await retrieve_and_generate(
                query="test",
                store=mock_store,
                top_k=10,
            )

        mock_store.search.assert_called_once_with(query="test", top_k=10)


class TestSearchResult:
    """Tests for SearchResult dataclass."""

    def test_search_result_creation(self):
        """Test SearchResult can be created with all fields."""
        result = SearchResult(
            id="doc1",
            document="Test document",
            score=0.95,
            metadata={"source": "test"},
        )

        assert result.id == "doc1"
        assert result.document == "Test document"
        assert result.score == 0.95
        assert result.metadata == {"source": "test"}
