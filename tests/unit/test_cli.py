"""Tests for eval CLI runner."""

import json
from unittest.mock import AsyncMock, patch

import pytest

from progression_labs.llm.eval.batch import BatchEvalResult
from progression_labs.llm.eval.cli import cli, main


class TestMain:
    """Tests for main() async function."""

    @pytest.mark.asyncio
    async def test_main_all_pass_returns_zero(self, tmp_path):
        """Exit code 0 when all tests pass."""
        test_data = {
            "test_cases": [
                {"input": "q", "output": "a", "expected": "a"},
            ],
            "metrics": ["answer_relevancy"],
            "threshold": 0.7,
        }
        test_file = tmp_path / "tests.json"
        test_file.write_text(json.dumps(test_data))

        mock_result = BatchEvalResult(
            total=1, passed=1, failed=0, pass_rate=1.0, results=[]
        )
        with patch(
            "progression_labs.llm.eval.cli.run_eval", new_callable=AsyncMock, return_value=mock_result
        ):
            exit_code = await main(str(test_file))

        assert exit_code == 0

    @pytest.mark.asyncio
    async def test_main_some_fail_returns_one(self, tmp_path):
        """Exit code 1 when any test fails."""
        test_data = {
            "test_cases": [
                {"input": "q", "output": "a", "expected": "a"},
            ],
            "metrics": ["answer_relevancy"],
            "threshold": 0.7,
        }
        test_file = tmp_path / "tests.json"
        test_file.write_text(json.dumps(test_data))

        mock_result = BatchEvalResult(
            total=2, passed=1, failed=1, pass_rate=0.5, results=[]
        )
        with patch(
            "progression_labs.llm.eval.cli.run_eval", new_callable=AsyncMock, return_value=mock_result
        ):
            exit_code = await main(str(test_file))

        assert exit_code == 1

    @pytest.mark.asyncio
    async def test_main_default_metrics_fallback(self, tmp_path):
        """Uses answer_relevancy when metrics not specified in JSON."""
        test_data = {
            "test_cases": [
                {"input": "q", "output": "a"},
            ],
        }
        test_file = tmp_path / "tests.json"
        test_file.write_text(json.dumps(test_data))

        mock_result = BatchEvalResult(
            total=1, passed=1, failed=0, pass_rate=1.0, results=[]
        )
        with patch(
            "progression_labs.llm.eval.cli.run_eval", new_callable=AsyncMock, return_value=mock_result
        ) as mock_run:
            await main(str(test_file))

        # Verify default metric was used (positional args)
        from progression_labs.llm.eval.metrics import Metric

        call_args = mock_run.call_args
        # run_eval(test_cases, metrics, threshold) — all positional
        assert call_args[0][1] == [Metric.ANSWER_RELEVANCY]

    @pytest.mark.asyncio
    async def test_main_json_output(self, tmp_path, capsys):
        """Prints JSON output with pass/fail stats."""
        test_data = {
            "test_cases": [{"input": "q", "output": "a"}],
        }
        test_file = tmp_path / "tests.json"
        test_file.write_text(json.dumps(test_data))

        mock_result = BatchEvalResult(
            total=2, passed=2, failed=0, pass_rate=1.0, results=[]
        )
        with patch(
            "progression_labs.llm.eval.cli.run_eval", new_callable=AsyncMock, return_value=mock_result
        ):
            await main(str(test_file), output_format="json")

        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["total"] == 2
        assert output["passed"] == 2
        assert output["failed"] == 0
        assert output["pass_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_main_custom_threshold(self, tmp_path):
        """Passes custom threshold from JSON to run_eval."""
        test_data = {
            "test_cases": [{"input": "q", "output": "a"}],
            "threshold": 0.9,
        }
        test_file = tmp_path / "tests.json"
        test_file.write_text(json.dumps(test_data))

        mock_result = BatchEvalResult(
            total=1, passed=1, failed=0, pass_rate=1.0, results=[]
        )
        with patch(
            "progression_labs.llm.eval.cli.run_eval", new_callable=AsyncMock, return_value=mock_result
        ) as mock_run:
            await main(str(test_file))

        # run_eval(test_cases, metrics, threshold) — threshold is 3rd positional
        assert mock_run.call_args[0][2] == 0.9


class TestCli:
    """Tests for cli() entry point."""

    def test_cli_calls_main_and_exits(self):
        """cli() parses args, calls main, and exits with its return code."""
        with (
            patch("sys.argv", ["eval", "tests.json"]),
            patch(
                "progression_labs.llm.eval.cli.main", new_callable=AsyncMock, return_value=0
            ) as mock_main,
            pytest.raises(SystemExit) as exc_info,
        ):
            cli()

        mock_main.assert_called_once_with("tests.json", "json")
        assert exc_info.value.code == 0

    def test_cli_exits_nonzero_on_failure(self):
        """cli() exits with code 1 when main returns 1."""
        with (
            patch("sys.argv", ["eval", "tests.json"]),
            patch(
                "progression_labs.llm.eval.cli.main", new_callable=AsyncMock, return_value=1
            ),
            pytest.raises(SystemExit) as exc_info,
        ):
            cli()

        assert exc_info.value.code == 1
