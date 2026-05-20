"""Composite evaluation suite — runs multiple evaluators and aggregates results."""

import json
from pathlib import Path
from typing import Optional

from remllm.eval.base import Evaluator, EvalReport
from remllm.eval.executable import ExecutableEvaluator
from remllm.eval.quality import QualityEvaluator


class EvaluationSuite:
    def __init__(self, evaluators: Optional[list[Evaluator]] = None):
        self.evaluators: list[Evaluator] = evaluators or [
            QualityEvaluator(),
            ExecutableEvaluator(),
        ]

    def add(self, evaluator: Evaluator) -> "EvaluationSuite":
        self.evaluators.append(evaluator)
        return self

    def run(self, model_name: str, rows: list[dict], **kwargs) -> dict[str, EvalReport]:
        results: dict[str, EvalReport] = {}
        for evaluator in self.evaluators:
            name = type(evaluator).__name__
            print(f"  Running {name}...")
            try:
                results[name] = evaluator.evaluate(model_name, rows, **kwargs)
            except Exception as exc:
                print(f"  WARNING: {name} failed: {exc}")
        return results

    def write_reports(
        self, results: dict[str, EvalReport], output_dir: Path, prefix: str = "eval"
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, report in results.items():
            path = output_dir / f"{prefix}_{name.lower()}.json"
            report.write(path)

    def summarize(self, results: dict[str, EvalReport]) -> dict:
        summary = {}
        for name, report in results.items():
            summary[name] = report.rates
        return summary


def run_full_evaluation(
    model_name: str,
    rows: list[dict],
    output_dir: Path,
    prefix: str = "eval",
    timeout_s: int = 30,
) -> dict:
    suite = EvaluationSuite()

    try:
        from remllm.eval.security_eval import SecurityEvaluator

        suite.add(SecurityEvaluator())
    except ImportError:
        pass

    try:
        from remllm.eval.beginner_eval import BeginnerEvaluator

        suite.add(BeginnerEvaluator())
    except ImportError:
        pass

    results = suite.run(model_name, rows, timeout_s=timeout_s)
    suite.write_reports(results, output_dir, prefix)
    summary = suite.summarize(results)
    print(json.dumps(summary, indent=2))
    return summary
