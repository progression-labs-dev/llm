# 06: DeepEval Testing

## Objective

Integrate DeepEval for LLM evaluation and testing in CI/CD pipelines.

## Tasks

- [ ] Create evaluation wrapper for common metrics
- [ ] Add pytest integration helpers
- [ ] Implement RAG-specific metrics
- [ ] Create CI-friendly evaluation runner
- [ ] Add custom metric support

## API Design

```python
from progression_labs.llm.eval import evaluate, run_eval, Metric

# Simple evaluation
result = await evaluate(
    input="What is 2+2?",
    output="4",
    expected="4",
    metrics=[Metric.ANSWER_RELEVANCY],
)
print(result.passed)  # True
print(result.scores)  # {"answer_relevancy": 0.95}

# Batch evaluation
results = await run_eval(
    test_cases=[
        {"input": "...", "output": "...", "expected": "..."},
        ...
    ],
    metrics=[Metric.ANSWER_RELEVANCY, Metric.FAITHFULNESS],
)
```

## Implementation

### Metrics Enum

```python
# src/progression_labs/llm/eval/metrics.py

from enum import Enum

class Metric(str, Enum):
    """Available evaluation metrics."""

    # General
    ANSWER_RELEVANCY = "answer_relevancy"
    FAITHFULNESS = "faithfulness"
    HALLUCINATION = "hallucination"
    TOXICITY = "toxicity"
    BIAS = "bias"

    # RAG-specific
    CONTEXTUAL_PRECISION = "contextual_precision"
    CONTEXTUAL_RECALL = "contextual_recall"
    CONTEXTUAL_RELEVANCY = "contextual_relevancy"
```

### Evaluation Functions

```python
# src/progression_labs/llm/eval/evaluate.py

from dataclasses import dataclass
from deepeval import evaluate as deepeval_evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
    ToxicityMetric,
    BiasMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
)

from .metrics import Metric


@dataclass
class EvalResult:
    """Result of an evaluation."""
    passed: bool
    scores: dict[str, float]
    details: dict[str, str]


def _get_metric_instance(metric: Metric, threshold: float = 0.7):
    """Convert Metric enum to DeepEval metric instance."""
    mapping = {
        Metric.ANSWER_RELEVANCY: AnswerRelevancyMetric(threshold=threshold),
        Metric.FAITHFULNESS: FaithfulnessMetric(threshold=threshold),
        Metric.HALLUCINATION: HallucinationMetric(threshold=threshold),
        Metric.TOXICITY: ToxicityMetric(threshold=threshold),
        Metric.BIAS: BiasMetric(threshold=threshold),
        Metric.CONTEXTUAL_PRECISION: ContextualPrecisionMetric(threshold=threshold),
        Metric.CONTEXTUAL_RECALL: ContextualRecallMetric(threshold=threshold),
        Metric.CONTEXTUAL_RELEVANCY: ContextualRelevancyMetric(threshold=threshold),
    }
    return mapping[metric]


async def evaluate(
    input: str,
    output: str,
    expected: str | None = None,
    context: list[str] | None = None,
    metrics: list[Metric] | None = None,
    threshold: float = 0.7,
) -> EvalResult:
    """
    Evaluate a single LLM output.

    Args:
        input: The input/query
        output: The LLM's actual output
        expected: Optional expected output
        context: Optional retrieved context (for RAG metrics)
        metrics: List of metrics to evaluate
        threshold: Score threshold for passing

    Returns:
        EvalResult with scores and pass/fail status
    """
    if metrics is None:
        metrics = [Metric.ANSWER_RELEVANCY]

    test_case = LLMTestCase(
        input=input,
        actual_output=output,
        expected_output=expected,
        retrieval_context=context,
    )

    metric_instances = [_get_metric_instance(m, threshold) for m in metrics]

    # Run evaluation
    results = deepeval_evaluate([test_case], metric_instances)

    # Aggregate results
    scores = {}
    details = {}
    all_passed = True

    for metric_result in results.test_results[0].metrics_data:
        scores[metric_result.name] = metric_result.score
        details[metric_result.name] = metric_result.reason
        if metric_result.score < threshold:
            all_passed = False

    return EvalResult(
        passed=all_passed,
        scores=scores,
        details=details,
    )
```

### Batch Evaluation

```python
# src/progression_labs/llm/eval/batch.py

from dataclasses import dataclass
from .evaluate import evaluate, EvalResult
from .metrics import Metric


@dataclass
class BatchEvalResult:
    """Results from batch evaluation."""
    total: int
    passed: int
    failed: int
    pass_rate: float
    results: list[EvalResult]


async def run_eval(
    test_cases: list[dict],
    metrics: list[Metric],
    threshold: float = 0.7,
) -> BatchEvalResult:
    """
    Run evaluation on multiple test cases.

    Args:
        test_cases: List of dicts with input, output, expected, context
        metrics: Metrics to evaluate
        threshold: Score threshold

    Returns:
        BatchEvalResult with aggregate statistics
    """
    results = []
    passed = 0

    for case in test_cases:
        result = await evaluate(
            input=case["input"],
            output=case["output"],
            expected=case.get("expected"),
            context=case.get("context"),
            metrics=metrics,
            threshold=threshold,
        )
        results.append(result)
        if result.passed:
            passed += 1

    return BatchEvalResult(
        total=len(test_cases),
        passed=passed,
        failed=len(test_cases) - passed,
        pass_rate=passed / len(test_cases) if test_cases else 0,
        results=results,
    )
```

### Pytest Integration

```python
# src/progression_labs/llm/eval/pytest_plugin.py

import pytest
from .evaluate import evaluate
from .metrics import Metric


def llm_test(
    input: str,
    output: str,
    expected: str | None = None,
    metrics: list[Metric] | None = None,
    threshold: float = 0.7,
):
    """
    Pytest assertion for LLM outputs.

    Usage:
        def test_my_llm():
            output = my_llm_function("What is 2+2?")
            llm_test(
                input="What is 2+2?",
                output=output,
                expected="4",
                metrics=[Metric.ANSWER_RELEVANCY],
            )
    """
    import asyncio
    result = asyncio.run(evaluate(
        input=input,
        output=output,
        expected=expected,
        metrics=metrics or [Metric.ANSWER_RELEVANCY],
        threshold=threshold,
    ))

    if not result.passed:
        pytest.fail(
            f"LLM evaluation failed:\n"
            f"Scores: {result.scores}\n"
            f"Details: {result.details}"
        )
```

## CLI Runner

```python
# src/progression_labs/llm/eval/cli.py

import json
import sys
from .batch import run_eval
from .metrics import Metric


async def main(test_file: str, output_format: str = "json"):
    """Run evaluations from a JSON test file."""
    with open(test_file) as f:
        data = json.load(f)

    test_cases = data["test_cases"]
    metrics = [Metric(m) for m in data.get("metrics", ["answer_relevancy"])]
    threshold = data.get("threshold", 0.7)

    result = await run_eval(test_cases, metrics, threshold)

    if output_format == "json":
        print(json.dumps({
            "total": result.total,
            "passed": result.passed,
            "failed": result.failed,
            "pass_rate": result.pass_rate,
        }))

    # Exit with non-zero if any failed (for CI)
    sys.exit(0 if result.failed == 0 else 1)
```

## Tests

```python
# tests/test_eval.py

import pytest
from progression_labs.llm.eval import evaluate, run_eval, Metric

@pytest.mark.asyncio
async def test_evaluate_relevancy():
    result = await evaluate(
        input="What is the capital of France?",
        output="Paris is the capital of France.",
        expected="Paris",
        metrics=[Metric.ANSWER_RELEVANCY],
    )
    assert result.passed
    assert result.scores["answer_relevancy"] > 0.7

@pytest.mark.asyncio
async def test_batch_eval():
    results = await run_eval(
        test_cases=[
            {"input": "2+2?", "output": "4", "expected": "4"},
            {"input": "Capital of UK?", "output": "London", "expected": "London"},
        ],
        metrics=[Metric.ANSWER_RELEVANCY],
    )
    assert results.pass_rate == 1.0
```

## CI Integration

```yaml
# .github/workflows/eval.yml
name: LLM Evaluation

on:
  push:
    branches: [main]
  pull_request:

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -e ".[dev]"
      - run: python -m progression_labs.llm.eval.cli tests/eval_cases.json
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

## Acceptance Criteria

- [ ] `evaluate()` scores single outputs
- [ ] All metric types work (relevancy, faithfulness, etc.)
- [ ] RAG metrics work with context
- [ ] `run_eval()` handles batch evaluation
- [ ] Pytest integration works
- [ ] CLI runner exits with proper codes for CI
- [ ] JSON output format works
