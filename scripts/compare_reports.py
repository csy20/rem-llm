#!/usr/bin/env python3
"""Thin CLI wrapper around remllm.eval.comparator — see `remllm compare`."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from remllm.eval.comparator import compare_reports

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare baseline and post-train reports."
    )
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--post", required=True)
    parser.add_argument("--baseline-exec", default="")
    parser.add_argument("--post-exec", default="")
    args = parser.parse_args()

    compare_reports(
        Path(args.baseline),
        Path(args.post),
        Path(args.baseline_exec) if args.baseline_exec else None,
        Path(args.post_exec) if args.post_exec else None,
    )
