#!/usr/bin/env python3
"""Thin CLI wrapper around remllm.train.unsloth — see `remllm train`."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from remllm.train.unsloth import train_unsloth

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QLoRA training with Unsloth.")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    train_unsloth(Path(args.config))
