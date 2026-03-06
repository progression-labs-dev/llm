"""CLI runner for LLM evaluation."""

import argparse
import asyncio
import json
import sys

from .batch import EvalTestCase, run_eval
from .metrics import Metric


async def main(test_file: str, output_format: str = "json") -> int:
    """Run evaluations from a JSON test file."""
    with open(test_file) as f:
        data = json.load(f)

    test_cases = [EvalTestCase(**tc) for tc in data["test_cases"]]
    metrics = [Metric(m) for m in data.get("metrics", ["answer_relevancy"])]
    threshold = data.get("threshold", 0.7)

    result = await run_eval(test_cases, metrics, threshold)

    if output_format == "json":
        print(
            json.dumps(
                {
                    "total": result.total,
                    "passed": result.passed,
                    "failed": result.failed,
                    "pass_rate": result.pass_rate,
                }
            )
        )

    # Exit with non-zero if any failed (for CI)
    return 0 if result.failed == 0 else 1


def cli():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run LLM evaluations")
    parser.add_argument("test_file", help="Path to JSON test file")
    parser.add_argument(
        "--format",
        choices=["json"],
        default="json",
        help="Output format",
    )
    args = parser.parse_args()

    exit_code = asyncio.run(main(args.test_file, args.format))
    sys.exit(exit_code)


if __name__ == "__main__":
    cli()
