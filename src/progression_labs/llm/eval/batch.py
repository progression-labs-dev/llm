"""Batch evaluation functions."""

from dataclasses import dataclass

from pydantic import BaseModel

from .evaluate import EvalResult, evaluate
from .metrics import Metric


class EvalTestCase(BaseModel):
    """A single test case for LLM evaluation."""

    input: str
    output: str
    expected: str | None = None
    context: list[str] | None = None


@dataclass
class BatchEvalResult:
    """Results from batch evaluation."""

    total: int
    passed: int
    failed: int
    pass_rate: float
    results: list[EvalResult]


async def run_eval(
    test_cases: list[EvalTestCase],
    metrics: list[Metric],
    threshold: float = 0.7,
) -> BatchEvalResult:
    """
    Run evaluation on multiple test cases.

    Args:
        test_cases: List of EvalTestCase instances
        metrics: Metrics to evaluate
        threshold: Score threshold

    Returns:
        BatchEvalResult with aggregate statistics
    """
    results = []
    passed = 0

    for case in test_cases:
        result = await evaluate(
            input=case.input,
            output=case.output,
            expected=case.expected,
            context=case.context,
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
