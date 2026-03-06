"""Core evaluation functions using DeepEval."""

from dataclasses import dataclass

from .metrics import Metric


@dataclass
class EvalResult:
    """Result of an evaluation."""

    passed: bool
    scores: dict[str, float]
    details: dict[str, str]


def _get_metric_instance(metric: Metric, threshold: float = 0.7):
    """Convert Metric enum to DeepEval metric instance."""
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        BiasMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
        HallucinationMetric,
        ToxicityMetric,
    )

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

    from deepeval.evaluate import evaluate as deepeval_evaluate
    from deepeval.test_case import LLMTestCase

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
    scores: dict[str, float] = {}
    details: dict[str, str] = {}
    all_passed = True

    metrics_data = results.test_results[0].metrics_data or []
    for metric_result in metrics_data:
        score = metric_result.score if metric_result.score is not None else 0.0
        scores[metric_result.name] = score
        details[metric_result.name] = metric_result.reason or ""
        if score < threshold:
            all_passed = False

    return EvalResult(
        passed=all_passed,
        scores=scores,
        details=details,
    )
