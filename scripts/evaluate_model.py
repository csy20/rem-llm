#!/usr/bin/env python3
"""Thin CLI wrapper around remllm.eval.quality — see `remllm eval quality`."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from remllm.data.loader import load_jsonl
from remllm.eval.quality import QualityEvaluator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on fixed eval set.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--timeout-s", type=int, default=None)
    args = parser.parse_args()

    import yaml

    config_path = Path(args.config)
    with config_path.open("r") as handle:
        config = yaml.safe_load(handle)
    rows = load_jsonl(config_path.parent.parent / config["data"]["eval_file"])

    if not rows:
        raise SystemExit(f"No eval rows found in {config['data']['eval_file']}")

    report = QualityEvaluator().evaluate(args.model, rows, timeout_s=args.timeout_s)
    report.write(Path(args.report))
    print(json.dumps(report.rates, indent=2))
