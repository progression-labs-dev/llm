# LLM Package Implementation Plan

## Overview

This document outlines the implementation plan for `@progression-labs/llm`, a unified Python LLM SDK.

## Components

| # | Component | Description | Dependency |
|---|-----------|-------------|------------|
| 1 | [Project Setup](./01-project-setup.md) | Package structure, dependencies, CI/CD | None |
| 2 | [LiteLLM Integration](./02-litellm-integration.md) | Unified LLM API wrapper | 1 |
| 3 | [Instructor Integration](./03-instructor-integration.md) | Structured output with Pydantic | 2 |
| 4 | [Langfuse Observability](./04-langfuse-observability.md) | LLM tracing, prompts, evals | 2 |
| 5 | [ChromaDB RAG](./05-chromadb-rag.md) | Simple vector search | 2 |
| 6 | [DeepEval Testing](./06-deepeval-testing.md) | LLM evaluation framework | 3 |

## Dependency Graph

```
┌─────────────────┐
│  1. Project     │
│     Setup       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  2. LiteLLM     │
│  Integration    │
└────────┬────────┘
         │
    ┌────┴────┬──────────┬──────────┐
    ▼         ▼          ▼          ▼
┌───────┐ ┌───────┐ ┌─────────┐ ┌───────┐
│ 3.    │ │ 4.    │ │ 5.      │ │ 6.    │
│Instruc│ │Langfu │ │ChromaDB │ │Deep   │
│  tor  │ │  se   │ │  RAG    │ │ Eval  │
└───────┘ └───────┘ └─────────┘ └───────┘
```

## Implementation Order

1. **Project Setup** - Foundation for everything
2. **LiteLLM Integration** - Core functionality
3. **Instructor Integration** - Depends on LiteLLM
4. **Langfuse Observability** - Can be done in parallel with 3
5. **ChromaDB RAG** - Can be done in parallel with 3
6. **DeepEval Testing** - Needs Instructor for structured test cases

## Success Criteria

- [ ] Package installable via `pip install progression-labs-llm`
- [ ] Unified API for OpenAI, Anthropic, Google
- [ ] All LLM calls return validated Pydantic models
- [ ] Traces visible in Langfuse
- [ ] Simple RAG working with ChromaDB
- [ ] Evals running in CI with DeepEval
