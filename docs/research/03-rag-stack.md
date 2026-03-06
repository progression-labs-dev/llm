# RAG Stack Research

## Recommendation: ChromaDB (start) → Qdrant (scale)

**Confidence: Medium-High**

For simple RAG (embed + retrieve), start with ChromaDB for speed. Migrate to Qdrant when filtering or scale becomes important.

## Requirements

- Simple embed + retrieve
- Cloud-only (no self-hosted models)
- Production-ready path

## Vector Database Comparison

### ChromaDB

**Best for: Prototyping, small-to-medium projects**

| Aspect | Details |
|--------|---------|
| **Setup** | 5 minutes, no Docker, no servers |
| **API** | Python-first, developer-friendly |
| **Scale** | Good for small-to-medium datasets |
| **Integration** | Tight LangChain/LlamaIndex support |
| **Hosting** | Embedded or cloud |

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("docs")

collection.add(
    documents=["doc1", "doc2"],
    ids=["id1", "id2"]
)

results = collection.query(
    query_texts=["search query"],
    n_results=5
)
```

**Limitation**: Have a migration path ready for production scale.

---

### Qdrant

**Best for: Production with metadata filtering**

| Aspect | Details |
|--------|---------|
| **Performance** | Written in Rust, excellent speed |
| **Filtering** | Rich metadata filtering (integrated into search) |
| **Scale** | Efficient up to ~10M vectors |
| **Hosting** | Self-hosted, Docker, or Qdrant Cloud |

```python
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance

client = QdrantClient(":memory:")  # or url="http://localhost:6333"

client.create_collection(
    collection_name="docs",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
)

client.upsert(
    collection_name="docs",
    points=[
        PointStruct(id=1, vector=[...], payload={"type": "article"}),
    ]
)

results = client.search(
    collection_name="docs",
    query_vector=[...],
    query_filter={"must": [{"key": "type", "match": {"value": "article"}}]},
    limit=5
)
```

**Limitation**: Performance degrades beyond 10M vectors.

---

### pgvector

**Best for: Teams already using PostgreSQL**

| Aspect | Details |
|--------|---------|
| **Integration** | Add vector search to existing Postgres |
| **SQL** | Combine vector and relational queries |
| **Cost** | Free extension, use existing infra |
| **Performance** | "Good enough" for many use cases |

```python
import psycopg2

# After installing pgvector extension
cur.execute("""
    CREATE TABLE documents (
        id SERIAL PRIMARY KEY,
        content TEXT,
        embedding vector(1536)
    );
    CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);
""")

# Query
cur.execute("""
    SELECT content, embedding <=> %s AS distance
    FROM documents
    ORDER BY embedding <=> %s
    LIMIT 5
""", (query_embedding, query_embedding))
```

**Limitation**: Not as performant as dedicated vector DBs; indexes use significant memory.

---

## Decision Framework

| Situation | Choice |
|-----------|--------|
| Starting out, simple RAG | ChromaDB |
| Need metadata filtering | Qdrant |
| Already have Postgres | pgvector |
| 10M+ vectors | Qdrant Cloud or Pinecone |

## Embedding Models

For cloud-only, use provider APIs:

| Provider | Model | Dimensions | Cost |
|----------|-------|------------|------|
| OpenAI | `text-embedding-3-small` | 1536 | $0.02/1M tokens |
| OpenAI | `text-embedding-3-large` | 3072 | $0.13/1M tokens |
| Google | `text-embedding-004` | 768 | $0.00001/1K chars |

LiteLLM supports embeddings via:

```python
from litellm import embedding

response = embedding(
    model="text-embedding-3-small",
    input=["Hello world"]
)
```

## Recommendation

1. **Start with ChromaDB** for development and initial deployment
2. **Design for portability** - abstract vector DB behind interface
3. **Migrate to Qdrant** when:
   - Need metadata filtering
   - Dataset grows beyond prototype
   - Need production SLAs

## Sources

- [Best Vector Databases 2025 - Firecrawl](https://www.firecrawl.dev/blog/best-vector-databases-2025)
- [Vector Database Comparison - LiquidMetal AI](https://liquidmetal.ai/casesAndBlogs/vector-comparison/)
- [How to Choose Your Vector Database](https://learnwithparam.com/blog/choosing-vector-database-chroma-qdrant-weaviate-pgvector-pinecone-vespa)
- [RAG from Scratch with Qdrant](https://techlife.blog/posts/implementing-rag-from-scratch-qdrant/)
