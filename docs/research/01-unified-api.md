# Unified LLM API Research

## Recommendation: LiteLLM

**Confidence: High**

LiteLLM is the clear choice for unified LLM API access across OpenAI, Anthropic, and Google.

## Overview

LiteLLM is a Python SDK and Proxy Server (AI Gateway) that provides a single OpenAI-compatible interface to 100+ LLM providers.

- **GitHub**: https://github.com/BerriAI/litellm
- **Docs**: https://docs.litellm.ai/docs/
- **PyPI**: https://pypi.org/project/litellm/
- **License**: MIT (fully open-source)
- **Latest Version**: Released January 21, 2026
- **Python**: Requires 3.9+

## Key Features

| Feature | Description |
|---------|-------------|
| **Unified Interface** | Single OpenAI-style API for all providers |
| **Cost Tracking** | Built-in token and cost tracking |
| **Fallback Routing** | Primary model fails → automatic backup |
| **Rate Limiting** | Built-in retry with backoff |
| **Streaming** | Full streaming support |
| **Async** | Async-first design |
| **Error Mapping** | All exceptions map to OpenAI exception types |

## Performance

- **8ms P95 latency** at 1,000 RPS
- Minimal overhead on top of provider APIs

## Supported Endpoints

- `/chat/completions`
- `/responses` (new OpenAI Responses API)
- `/embeddings`
- `/images`
- `/audio`
- `/batches`
- `/rerank`
- `/messages` (Anthropic native)

## Provider Support

All three required providers are fully supported:

| Provider | Status | Notes |
|----------|--------|-------|
| OpenAI | Full | Including GPT-5 family with reasoning_effort |
| Anthropic | Full | Native /v1/messages support |
| Google | Full | Gemini including Realtime API |

## Usage Example

```python
from litellm import completion

# OpenAI
response = completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}]
)

# Anthropic
response = completion(
    model="claude-sonnet-4-20250514",
    messages=[{"role": "user", "content": "Hello"}]
)

# Google
response = completion(
    model="gemini/gemini-2.0-flash",
    messages=[{"role": "user", "content": "Hello"}]
)
```

## Async Support

```python
from litellm import acompletion

response = await acompletion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}]
)
```

## Fallback Configuration

```python
from litellm import completion

response = completion(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Hello"}],
    fallbacks=["claude-sonnet-4-20250514", "gemini/gemini-2.0-flash"]
)
```

## Cost Tracking

```python
from litellm import completion_cost

response = completion(model="gpt-4o", messages=[...])
cost = completion_cost(response)
```

## Alternatives Considered

| Tool | Why Not |
|------|---------|
| Raw SDKs | Would need 3 separate implementations |
| LangChain | Too heavy for just API abstraction |
| Custom wrapper | Maintenance burden, no cost tracking |

## Integration Notes

- LiteLLM works seamlessly with Instructor for structured output
- Logging can be configured to output to Better Stack
- Error handling follows OpenAI patterns (easy to migrate existing code)

## Sources

- [GitHub - BerriAI/litellm](https://github.com/BerriAI/litellm)
- [LiteLLM Documentation](https://docs.litellm.ai/docs/)
- [LiteLLM Changelog](https://www.litellm.ai/changelog)
- [LiteLLM Guide - DataCamp](https://www.datacamp.com/tutorial/litellm)
