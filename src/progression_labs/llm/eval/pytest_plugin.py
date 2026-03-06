"""Pytest integration for LLM evaluation."""

import asyncio

import pytest

from .evaluate import evaluate
from .metrics import Metric


def llm_test(
    input: str,
    output: str,
    expected: str | None = None,
    context: list[str] | None = None,
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
    result = asyncio.run(
        evaluate(
            input=input,
            output=output,
            expected=expected,
            context=context,
            metrics=metrics or [Metric.ANSWER_RELEVANCY],
            threshold=threshold,
        )
    )

    if not result.passed:
        pytest.fail(f"LLM evaluation failed:\nScores: {result.scores}\nDetails: {result.details}")
