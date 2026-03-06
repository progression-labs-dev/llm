# Observability Research

## Recommendation: Langfuse (LLM) + SigNoz (General)

**Confidence: High**

Use Langfuse for LLM-specific observability (prompt versioning, evals, cost tracking). Use SigNoz for general application logs, traces, and metrics.

## Requirements

- Cost tracking (tokens, spend per request)
- Latency monitoring
- Model usage tracking
- Traces, metrics, logs → **SigNoz**

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Your Application                        │
├─────────────────────────────────────────────────────────────┤
│  LiteLLM + Instructor                                        │
│       │                                                      │
│       ├─────────────────────────────────────┐               │
│       ▼                                     ▼               │
│  ┌──────────────┐                   ┌──────────────┐        │
│  │   Langfuse   │                   │    SigNoz    │        │
│  │  (LLM-specific)                  │   (general)  │        │
│  │              │                   │              │        │
│  │ • Prompts    │                   │ • App logs   │        │
│  │ • Evals      │                   │ • Traces     │        │
│  │ • LLM costs  │                   │ • Metrics    │        │
│  │ • Traces     │                   │              │        │
│  └──────────────┘                   └──────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

---

## SigNoz (Primary)

SigNoz is an open-source, OpenTelemetry-native observability platform for traces, metrics, and logs.

- **Docs**: https://signoz.io/docs/
- **LiteLLM Guide**: https://signoz.io/docs/litellm-observability/
- **License**: Open-source (self-hostable)

### Why SigNoz

| Feature | Details |
|---------|---------|
| **OpenTelemetry Native** | Built on OTel from the ground up |
| **Unified Platform** | Traces + metrics + logs in one place |
| **LLM Support** | Native LiteLLM, LangChain, LlamaIndex instrumentation |
| **Self-Hostable** | Full control over data |
| **Cost** | Open-source, pay only for infrastructure |

### LiteLLM + SigNoz Integration

SigNoz provides direct LiteLLM observability via OpenTelemetry:

```python
# Install dependencies
# pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
# pip install opentelemetry-instrumentation-openai  # or use openllmetry

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# Configure OTel to export to SigNoz
trace.set_tracer_provider(TracerProvider())
otlp_exporter = OTLPSpanExporter(
    endpoint="https://ingest.signoz.io:443",  # or self-hosted
    headers={"signoz-ingestion-key": "<your-key>"}
)
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(otlp_exporter)
)
```

### OpenLLMetry (Traceloop)

OpenLLMetry provides automatic LLM instrumentation that exports to any OTel backend (including SigNoz):

```python
# pip install traceloop-sdk

from traceloop.sdk import Traceloop

Traceloop.init(
    app_name="progression-labs-llm",
    api_endpoint="https://ingest.signoz.io:443",
    headers={"signoz-ingestion-key": "<your-key>"}
)

# Now all LiteLLM calls are automatically traced
from litellm import completion
response = completion(model="gpt-4o", messages=[...])
# ^ Automatically creates spans with token counts, latency, model info
```

### What You Get in SigNoz

- **Traces**: Every LLM call as a span with duration, tokens, model
- **Metrics**: Latency percentiles, token usage over time, error rates
- **Logs**: Structured logs with correlation IDs
- **Dashboards**: Build custom dashboards for LLM costs

---

## Langfuse (Optional - LLM-Specific UI)

Langfuse is purpose-built for LLM observability with features SigNoz doesn't have.

- **Website**: https://langfuse.com/
- **GitHub**: https://github.com/langfuse/langfuse (19k+ stars)
- **License**: MIT (self-hostable)
- **Free Tier**: 50k observations/month

### When to Add Langfuse

| Need | SigNoz | Langfuse |
|------|--------|----------|
| Basic traces & metrics | Yes | Yes |
| Prompt versioning | No | **Yes** |
| Prompt playground | No | **Yes** |
| LLM-as-judge evals | No | **Yes** |
| Dataset management | No | **Yes** |
| User feedback tracking | No | **Yes** |
| Cost analytics dashboard | Manual | **Built-in** |

### Langfuse + LiteLLM Integration

```python
import litellm

# Enable Langfuse callback (alongside OTel for SigNoz)
litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]

# Environment variables
# LANGFUSE_PUBLIC_KEY=pk-...
# LANGFUSE_SECRET_KEY=sk-...
# LANGFUSE_HOST=https://cloud.langfuse.com (or self-hosted)
```

### Langfuse Python Decorator

```python
from langfuse.decorators import observe, langfuse_context

@observe()
async def generate_response(prompt: str):
    response = await completion(model="gpt-4o", messages=[...])

    # Add custom metadata
    langfuse_context.update_current_observation(
        metadata={"user_id": "123", "feature": "chat"}
    )

    return response
```

### Langfuse Key Features

1. **Prompt Management**
   - Version control prompts
   - A/B test prompt variants
   - Rollback to previous versions

2. **Evaluations**
   - LLM-as-judge scoring
   - Human feedback collection
   - Custom evaluation functions

3. **Datasets**
   - Create test datasets from production traces
   - Run batch evaluations
   - Track regression over time

---

## Arize Phoenix (Optional - Local Development)

Phoenix is great for local development and embeddings analysis.

- **GitHub**: https://github.com/Arize-ai/phoenix
- **License**: Open-source
- **Runs**: Locally in notebook or as server

### When to Use Phoenix

| Use Case | Phoenix |
|----------|---------|
| Local development | Excellent |
| Embeddings visualization | **Best-in-class** |
| Drift detection | Yes |
| Production monitoring | Use SigNoz instead |

### Phoenix Setup

```python
# pip install arize-phoenix

import phoenix as px

# Launch local UI
session = px.launch_app()

# Instrument LiteLLM
from openinference.instrumentation.litellm import LiteLLMInstrumentor
LiteLLMInstrumentor().instrument()

# View traces at http://localhost:6006
```

### Phoenix vs Langfuse vs SigNoz

| Aspect | SigNoz | Langfuse | Phoenix |
|--------|--------|----------|---------|
| **Primary Use** | Production observability | LLM lifecycle | Local dev |
| **Traces** | Yes | Yes | Yes |
| **Metrics** | Yes | Limited | Limited |
| **Logs** | Yes | No | No |
| **Prompt Versioning** | No | Yes | No |
| **Embeddings Viz** | No | No | **Yes** |
| **Self-Host** | Yes | Yes | Yes (local) |
| **OTel Native** | Yes | Yes | Yes |

---

## Recommended Setup

### Production

```
LiteLLM → Langfuse (LLM observability)
        → SigNoz (general app observability)
```

| Tool | Responsibility |
|------|----------------|
| **Langfuse** | LLM traces, prompt versioning, evals, cost tracking |
| **SigNoz** | Application logs, general traces, infrastructure metrics |

### Why Both?

- **Langfuse** has purpose-built UI for LLM workflows (prompt playground, eval dashboards, cost analytics)
- **SigNoz** handles everything else (API latency, error rates, system metrics)
- Clean separation: LLM concerns vs. general observability

### Development (Optional)

```
LiteLLM → Phoenix (local)
```

- Fast local iteration
- Embeddings visualization for RAG debugging
- No cloud setup needed

---

## Implementation

```python
# progression_labs/llm/observability.py

import litellm
import os

def setup_observability():
    """Initialize observability for LLM calls."""

    # Langfuse for LLM-specific observability
    # Requires: LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]

    # Optional: Add custom callback for SigNoz general logging
    # (SigNoz integration handled separately via OpenTelemetry in main app)
```

### Environment Variables

```bash
# Langfuse (required)
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com  # or self-hosted URL

# SigNoz (configured in main app, not LLM package)
SIGNOZ_ENDPOINT=https://ingest.signoz.io:443
SIGNOZ_KEY=...
```

---

## Metrics to Track

| Metric | Source | Dashboard |
|--------|--------|-----------|
| **Latency (p50, p95, p99)** | OTel spans | SigNoz |
| **Token usage** | OTel attributes | SigNoz |
| **Cost per request** | Calculated | SigNoz |
| **Error rate** | OTel spans | SigNoz |
| **Model distribution** | OTel attributes | SigNoz |
| **Prompt performance** | Langfuse | Langfuse |

---

## Sources

- [SigNoz LiteLLM Observability](https://signoz.io/docs/litellm-observability/)
- [SigNoz LLM Observability with OpenTelemetry](https://signoz.io/blog/llm-observability-opentelemetry/)
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry)
- [Langfuse Documentation](https://langfuse.com/docs/observability/overview)
- [Arize Phoenix](https://github.com/Arize-ai/phoenix)
