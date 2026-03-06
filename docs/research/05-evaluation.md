# Evaluation Research

## Recommendation: DeepEval

**Confidence: High**

DeepEval is the pytest for LLMs - perfect for CI/CD integration and comprehensive evaluation.

## Overview

DeepEval is an open-source LLM evaluation framework for testing LLM outputs. It provides 50+ metrics, CI/CD integration, and LLM-as-judge evaluation.

- **Website**: https://deepeval.com/
- **GitHub**: https://github.com/confident-ai/deepeval
- **Docs**: https://deepeval.com/docs/getting-started
- **Python**: Requires 3.9+

## Key Features

| Feature | Description |
|---------|-------------|
| **50+ Metrics** | Research-backed evaluation metrics |
| **LLM-as-Judge** | Uses LLMs to evaluate LLM outputs |
| **CI/CD Ready** | JSON output, exit codes for pipelines |
| **Multi-Modal** | Supports text, images, and more |
| **RAG Metrics** | Specific metrics for retrieval systems |
| **Agent Metrics** | Six metrics for agentic workflows |

## Installation

```bash
pip install deepeval
```

## Usage Example

```python
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

test_case = LLMTestCase(
    input="What is the capital of France?",
    actual_output="Paris is the capital of France.",
    expected_output="Paris"
)

metric = AnswerRelevancyMetric()
evaluate([test_case], [metric])
```

## Pytest Integration

```python
# test_llm.py
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

def test_answer_relevancy():
    test_case = LLMTestCase(
        input="What is 2+2?",
        actual_output="4"
    )
    metric = AnswerRelevancyMetric(threshold=0.7)
    assert_test(test_case, [metric])
```

Run with:
```bash
deepeval test run test_llm.py
```

## Key Metrics

### General Metrics

| Metric | Purpose |
|--------|---------|
| `AnswerRelevancyMetric` | Is the answer relevant to the question? |
| `FaithfulnessMetric` | Is the answer grounded in the context? |
| `HallucinationMetric` | Does the output contain hallucinations? |
| `ToxicityMetric` | Is the output toxic or harmful? |
| `BiasMetric` | Does the output show bias? |

### RAG Metrics

| Metric | Purpose |
|--------|---------|
| `ContextualPrecisionMetric` | Are retrieved contexts relevant? |
| `ContextualRecallMetric` | Are all relevant contexts retrieved? |
| `ContextualRelevancyMetric` | Is the context useful for answering? |

### Agent Metrics

| Metric | Purpose |
|--------|---------|
| `TaskCompletionMetric` | Did the agent complete the task? |
| `ToolCorrectnessMetric` | Did the agent use tools correctly? |

## CI/CD Integration

```yaml
# .github/workflows/eval.yml
name: LLM Evaluation

on: [push]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install deepeval
      - run: deepeval test run tests/ --output-format json
```

## G-Eval

DeepEval uses G-Eval for flexible evaluation:

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

correctness = GEval(
    name="Correctness",
    criteria="Determine if the actual output is factually correct.",
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT
    ]
)
```

## Provider Support

DeepEval works with multiple LLM providers for evaluation:
- OpenAI
- Anthropic
- Google Gemini
- Ollama (local)
- Azure OpenAI

## Alternatives Considered

| Tool | Why Not |
|------|---------|
| RAGAS | RAG-specific, DeepEval is more general |
| TruLens | Less CI/CD focused |
| Braintrust | Managed platform, less open |
| Manual testing | Not scalable |

## Integration Notes

- Use `deepeval test run` in CI pipelines
- JSON output can be parsed for dashboards
- Metrics can be customized with G-Eval
- Works with any LLM output (not tied to specific framework)

## Sources

- [DeepEval Documentation](https://deepeval.com/docs/getting-started)
- [GitHub - confident-ai/deepeval](https://github.com/confident-ai/deepeval)
- [DeepEval Metrics Guide](https://deepeval.com/docs/metrics-introduction)
- [DeepEval Tutorial - DataCamp](https://www.datacamp.com/tutorial/deepeval)
