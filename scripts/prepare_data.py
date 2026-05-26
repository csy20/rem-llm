#!/usr/bin/env python3
"""Thin CLI wrapper around remllm.data.prepper — see `remllm data prepare`."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from remllm.data.prepper import prepare_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare train/val/eval datasets.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepare_data(Path(args.config), force=args.force)
