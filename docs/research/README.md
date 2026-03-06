# LLM Package Research

Research documentation for the `@progression-labs/llm` Python package.

## Requirements Summary

Based on initial discovery:

| Requirement | Decision |
|-------------|----------|
| **Providers** | OpenAI, Anthropic, Google (all three) |
| **Local Models** | Not required (cloud only) |
| **RAG Complexity** | Simple (embed + retrieve) |
| **Structured Output** | Critical (all calls return Pydantic models) |

## Research Areas

| Area | File | Status |
|------|------|--------|
| Unified LLM API | [01-unified-api.md](./01-unified-api.md) | Complete |
| Structured Output | [02-structured-output.md](./02-structured-output.md) | Complete |
| RAG Stack | [03-rag-stack.md](./03-rag-stack.md) | Complete |
| Observability | [04-observability.md](./04-observability.md) | Complete |
| Evaluation | [05-evaluation.md](./05-evaluation.md) | Complete |

## Recommended Stack

```
┌───────────────────────────────────────────────────────────┐
│                    @progression-labs/llm                       │
├───────────────────────────────────────────────────────────┤
│  LiteLLM          │  Unified API for all providers        │
│  Instructor       │  Structured output + validation       │
│  ChromaDB         │  Vector storage for simple RAG        │
│  DeepEval         │  Testing and evaluation               │
│  Langfuse         │  LLM observability, prompts, evals    │
├───────────────────────────────────────────────────────────┤
│  SigNoz           │  General app logs, traces, metrics    │
└───────────────────────────────────────────────────────────┘
```

## Final Recommendations

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `litellm` | latest | Unified LLM API (OpenAI, Anthropic, Google) |
| `instructor` | latest | Structured output with Pydantic validation |
| `chromadb` | latest | Simple vector DB for RAG |
| `deepeval` | latest | LLM testing and evaluation |
| `langfuse` | latest | LLM observability, prompt versioning, evals |

### Key Design Decisions

1. **LiteLLM as the foundation**
   - Single interface for all providers
   - Built-in cost tracking, fallbacks, retries
   - 8ms P95 latency overhead

2. **Instructor for all LLM calls**
   - Every call returns a validated Pydantic model
   - Automatic retries on validation failure
   - Works with LiteLLM via `instructor.from_litellm()`

3. **ChromaDB for RAG (start simple)**
   - 5-minute setup, no infrastructure
   - Migrate to Qdrant when filtering/scale needed
   - Abstract behind interface for portability

4. **Langfuse for LLM observability, SigNoz for general**
   - Langfuse: LLM traces, prompt versioning, evals, cost tracking
   - SigNoz: Application logs, general traces, infrastructure metrics
   - Clean separation of concerns

5. **DeepEval for testing**
   - Pytest-style LLM testing
   - CI/CD ready with JSON output
   - RAG and agent-specific metrics

### Architecture

```python
# High-level usage pattern

from progression_labs.llm import complete, embed, evaluate

# Structured completion (all calls return Pydantic models)
result = await complete(
    prompt="Extract user info: John is 25",
    response_model=User,
    model="gpt-4o",  # or "claude-sonnet-4", "gemini-2.0-flash"
)

# Simple RAG
docs = await embed(texts=["doc1", "doc2"])
results = await search(query="find relevant", top_k=5)

# Evaluation
eval_results = evaluate(
    test_cases=[...],
    metrics=["relevancy", "faithfulness"]
)
```

### Next Steps

1. Set up Python package structure
2. Implement core LiteLLM + Instructor wrapper
3. Add observability callbacks
4. Build simple RAG interface
5. Add DeepEval integration
6. Write tests and documentation
