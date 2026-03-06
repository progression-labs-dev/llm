"""Tests for LLM evaluation module."""

from unittest.mock import MagicMock, patch

import pytest

from progression_labs.llm.eval import (
    BatchEvalResult,
    EvalResult,
    EvalTestCase,
    Metric,
    evaluate,
    llm_test,
    run_eval,
)


class TestMetric:
    """Tests for Metric enum."""

    def test_metric_values(self):
        """Test all metric values are defined."""
        assert Metric.ANSWER_RELEVANCY == "answer_relevancy"
        assert Metric.FAITHFULNESS == "faithfulness"
        assert Metric.HALLUCINATION == "hallucination"
        assert Metric.TOXICITY == "toxicity"
        assert Metric.BIAS == "bias"
        assert Metric.CONTEXTUAL_PRECISION == "contextual_precision"
        assert Metric.CONTEXTUAL_RECALL == "contextual_recall"
        assert Metric.CONTEXTUAL_RELEVANCY == "contextual_relevancy"

    def test_metric_from_string(self):
        """Test Metric can be created from string value."""
        assert Metric("answer_relevancy") == Metric.ANSWER_RELEVANCY


class TestEvalResult:
    """Tests for EvalResult dataclass."""

    def test_eval_result_creation(self):
        """Test EvalResult can be created with all fields."""
        result = EvalResult(
            passed=True,
            scores={"answer_relevancy": 0.95},
            details={"answer_relevancy": "The answer is relevant"},
        )

        assert result.passed is True
        assert result.scores["answer_relevancy"] == 0.95
        assert "relevant" in result.details["answer_relevancy"]


class TestEvaluate:
    """Tests for evaluate function."""

    @pytest.mark.asyncio
    async def test_evaluate_returns_result(self):
        """Test evaluate returns EvalResult with scores."""
        mock_metric_data = MagicMock()
        mock_metric_data.name = "answer_relevancy"
        mock_metric_data.score = 0.95
        mock_metric_data.reason = "The answer is relevant to the question"

        mock_test_result = MagicMock()
        mock_test_result.metrics_data = [mock_metric_data]

        mock_eval_result = MagicMock()
        mock_eval_result.test_results = [mock_test_result]

        with (
            patch(
                "progression_labs.llm.eval.evaluate._get_metric_instance",
                return_value=MagicMock(),
            ),
            patch(
                "deepeval.evaluate.evaluate",
                return_value=mock_eval_result,
            ),
        ):
            result = await evaluate(
                input="What is the capital of France?",
                output="Paris is the capital of France.",
                expected="Paris",
                metrics=[Metric.ANSWER_RELEVANCY],
            )

        assert isinstance(result, EvalResult)
        assert result.passed is True
        assert result.scores["answer_relevancy"] == 0.95
        assert "relevant" in result.details["answer_relevancy"]

    @pytest.mark.asyncio
    async def test_evaluate_fails_below_threshold(self):
        """Test evaluate marks as failed when below threshold."""
        mock_metric_data = MagicMock()
        mock_metric_data.name = "answer_relevancy"
        mock_metric_data.score = 0.5
        mock_metric_data.reason = "The answer is not relevant"

        mock_test_result = MagicMock()
        mock_test_result.metrics_data = [mock_metric_data]

        mock_eval_result = MagicMock()
        mock_eval_result.test_results = [mock_test_result]

        with (
            patch(
                "progression_labs.llm.eval.evaluate._get_metric_instance",
                return_value=MagicMock(),
            ),
            patch(
                "deepeval.evaluate.evaluate",
                return_value=mock_eval_result,
            ),
        ):
            result = await evaluate(
                input="What is the capital of France?",
                output="I don't know",
                metrics=[Metric.ANSWER_RELEVANCY],
                threshold=0.7,
            )

        assert result.passed is False
        assert result.scores["answer_relevancy"] == 0.5

    @pytest.mark.asyncio
    async def test_evaluate_with_context(self):
        """Test evaluate with RAG context."""
        mock_metric_data = MagicMock()
        mock_metric_data.name = "contextual_relevancy"
        mock_metric_data.score = 0.9
        mock_metric_data.reason = "Context is relevant"

        mock_test_result = MagicMock()
        mock_test_result.metrics_data = [mock_metric_data]

        mock_eval_result = MagicMock()
        mock_eval_result.test_results = [mock_test_result]

        with (
            patch(
                "progression_labs.llm.eval.evaluate._get_metric_instance",
                return_value=MagicMock(),
            ),
            patch(
                "deepeval.evaluate.evaluate",
                return_value=mock_eval_result,
            ),
        ):
            result = await evaluate(
                input="What is the capital of France?",
                output="Paris",
                context=["France is a country in Europe. Paris is its capital."],
                metrics=[Metric.CONTEXTUAL_RELEVANCY],
            )

        assert result.passed is True
        assert "contextual_relevancy" in result.scores

    @pytest.mark.asyncio
    async def test_evaluate_default_metric(self):
        """Test evaluate uses answer_relevancy as default metric."""
        mock_metric_data = MagicMock()
        mock_metric_data.name = "answer_relevancy"
        mock_metric_data.score = 0.85
        mock_metric_data.reason = "Relevant"

        mock_test_result = MagicMock()
        mock_test_result.metrics_data = [mock_metric_data]

        mock_eval_result = MagicMock()
        mock_eval_result.test_results = [mock_test_result]

        with (
            patch(
                "progression_labs.llm.eval.evaluate._get_metric_instance",
                return_value=MagicMock(),
            ),
            patch(
                "deepeval.evaluate.evaluate",
                return_value=mock_eval_result,
            ) as mock_deepeval,
        ):
            await evaluate(
                input="test",
                output="test response",
            )

        # Verify it was called (default metric should be used)
        mock_deepeval.assert_called_once()


class TestRunEval:
    """Tests for run_eval batch evaluation function."""

    @pytest.mark.asyncio
    async def test_run_eval_batch(self):
        """Test run_eval processes multiple test cases."""
        mock_metric_data = MagicMock()
        mock_metric_data.name = "answer_relevancy"
        mock_metric_data.score = 0.9
        mock_metric_data.reason = "Relevant"

        mock_test_result = MagicMock()
        mock_test_result.metrics_data = [mock_metric_data]

        mock_eval_result = MagicMock()
        mock_eval_result.test_results = [mock_test_result]

        with (
            patch(
                "progression_labs.llm.eval.evaluate._get_metric_instance",
                return_value=MagicMock(),
            ),
            patch(
                "deepeval.evaluate.evaluate",
                return_value=mock_eval_result,
            ),
        ):
            results = await run_eval(
                test_cases=[
                    EvalTestCase(input="2+2?", output="4", expected="4"),
                    EvalTestCase(input="Capital of UK?", output="London", expected="London"),
                ],
                metrics=[Metric.ANSWER_RELEVANCY],
            )

        assert isinstance(results, BatchEvalResult)
        assert results.total == 2
        assert results.passed == 2
        assert results.failed == 0
        assert results.pass_rate == 1.0
        assert len(results.results) == 2

    @pytest.mark.asyncio
    async def test_run_eval_mixed_results(self):
        """Test run_eval handles mixed pass/fail results."""
        call_count = 0

        def mock_deepeval(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            mock_metric_data = MagicMock()
            mock_metric_data.name = "answer_relevancy"
            # First call passes, second fails
            mock_metric_data.score = 0.9 if call_count == 1 else 0.3
            mock_metric_data.reason = "reason"

            mock_test_result = MagicMock()
            mock_test_result.metrics_data = [mock_metric_data]

            mock_eval_result = MagicMock()
            mock_eval_result.test_results = [mock_test_result]
            return mock_eval_result

        with (
            patch(
                "progression_labs.llm.eval.evaluate._get_metric_instance",
                return_value=MagicMock(),
            ),
            patch(
                "deepeval.evaluate.evaluate",
                side_effect=mock_deepeval,
            ),
        ):
            results = await run_eval(
                test_cases=[
                    EvalTestCase(input="good", output="good answer"),
                    EvalTestCase(input="bad", output="bad answer"),
                ],
                metrics=[Metric.ANSWER_RELEVANCY],
            )

        assert results.total == 2
        assert results.passed == 1
        assert results.failed == 1
        assert results.pass_rate == 0.5

    @pytest.mark.asyncio
    async def test_run_eval_empty_cases(self):
        """Test run_eval handles empty test cases."""
        results = await run_eval(
            test_cases=[],
            metrics=[Metric.ANSWER_RELEVANCY],
        )

        assert results.total == 0
        assert results.passed == 0
        assert results.failed == 0
        assert results.pass_rate == 0


class TestLlmTest:
    """Tests for pytest integration."""

    def test_llm_test_passes(self):
        """Test llm_test passes when evaluation passes."""
        mock_metric_data = MagicMock()
        mock_metric_data.name = "answer_relevancy"
        mock_metric_data.score = 0.9
        mock_metric_data.reason = "Relevant"

        mock_test_result = MagicMock()
        mock_test_result.metrics_data = [mock_metric_data]

        mock_eval_result = MagicMock()
        mock_eval_result.test_results = [mock_test_result]

        with (
            patch(
                "progression_labs.llm.eval.evaluate._get_metric_instance",
                return_value=MagicMock(),
            ),
            patch(
                "deepeval.evaluate.evaluate",
                return_value=mock_eval_result,
            ),
        ):
            # Should not raise
            llm_test(
                input="What is 2+2?",
                output="4",
                expected="4",
                metrics=[Metric.ANSWER_RELEVANCY],
            )

    def test_llm_test_fails(self):
        """Test llm_test fails when evaluation fails."""
        mock_metric_data = MagicMock()
        mock_metric_data.name = "answer_relevancy"
        mock_metric_data.score = 0.3
        mock_metric_data.reason = "Not relevant"

        mock_test_result = MagicMock()
        mock_test_result.metrics_data = [mock_metric_data]

        mock_eval_result = MagicMock()
        mock_eval_result.test_results = [mock_test_result]

        with (
            patch(
                "progression_labs.llm.eval.evaluate._get_metric_instance",
                return_value=MagicMock(),
            ),
            patch(
                "deepeval.evaluate.evaluate",
                return_value=mock_eval_result,
            ),
            pytest.raises(pytest.fail.Exception) as exc_info,
        ):
            llm_test(
                input="What is 2+2?",
                output="banana",
                expected="4",
                metrics=[Metric.ANSWER_RELEVANCY],
            )

        assert "LLM evaluation failed" in str(exc_info.value)
