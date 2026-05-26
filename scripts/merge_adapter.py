#!/usr/bin/env python3
"""Thin CLI wrapper around remllm.export.merge — see `remllm merge`."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from remllm.export.merge import merge_adapter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    merge_adapter(Path(args.config))
