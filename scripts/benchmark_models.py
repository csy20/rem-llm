#!/usr/bin/env python3
"""Thin CLI wrapper around remllm.eval.benchmark — see `remllm eval benchmark`."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from remllm.eval.benchmark import benchmark_models

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark Ollama model latency/throughput."
    )
    parser.add_argument("--models", required=True, help="Comma-separated model names.")
    parser.add_argument("--eval-file", default="data/eval.jsonl")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--report", default="models/evals/benchmark.json")
    args = parser.parse_args()

    model_names = [n.strip() for n in args.models.split(",") if n.strip()]
    if not model_names:
        raise SystemExit("No model names provided")

    report = benchmark_models(
        model_names, Path(args.eval_file), args.max_samples, args.timeout_s
    )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote benchmark report: {report_path}")
    print(json.dumps(report["models"], indent=2))
