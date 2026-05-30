"""Data preparation: train/val/eval splitting and fingerprint-aware caching."""

import argparse
import random
from pathlib import Path

from remllm.config import load_config_dict, resolve_project_root
from remllm.data.loader import (
    build_dataset_fingerprint,
    count_lines,
    load_cache,
    load_jsonl,
    validate_rows,
    write_cache,
    write_jsonl,
)
from remllm.logging import get_logger


def prepare_data(config_path: Path, force: bool = False) -> None:
    config = load_config_dict(config_path)
    root = resolve_project_root(config_path, str(config["data"]["raw_file"]))
    data_cfg = config["data"]
    seed = int(config["project"]["seed"])
    train_split = float(data_cfg["train_split"])

    raw_path = root / data_cfg["raw_file"]
    train_path = root / data_cfg["train_file"]
    val_path = root / data_cfg["val_file"]
    eval_path = root / data_cfg["eval_file"]
    cache_path = root / "data" / "prepare_cache.json"

    log = get_logger(phase="data_prep", raw_file=str(raw_path))

    dataset_fingerprint = build_dataset_fingerprint(raw_path, seed, train_split)
    cache_payload = load_cache(cache_path)
    outputs_exist = train_path.exists() and val_path.exists() and eval_path.exists()

    if (
        not force
        and outputs_exist
        and cache_payload.get("fingerprint") == dataset_fingerprint
    ):
        log.info(
            "data_prep_cache_hit",
            fingerprint=dataset_fingerprint["raw_sha256"],
            train_rows=count_lines(train_path),
            val_rows=count_lines(val_path),
            eval_rows=count_lines(eval_path),
        )
        return

    raw_rows = load_jsonl(raw_path)
    cleaned_rows, dropped = validate_rows(raw_rows)

    random.Random(seed).shuffle(cleaned_rows)
    if len(cleaned_rows) < 2:
        raise ValueError(
            f"Need at least 2 valid rows for train/val split, got {len(cleaned_rows)}"
        )
    split_idx = max(1, int(len(cleaned_rows) * train_split))
    split_idx = min(split_idx, len(cleaned_rows) - 1)

    train_rows = cleaned_rows[:split_idx]
    val_rows = cleaned_rows[split_idx:]

    eval_size = min(max(1, int(len(cleaned_rows) * 0.05)), 200)
    eval_rows = val_rows[:eval_size]

    write_jsonl(train_path, train_rows)
    write_jsonl(val_path, val_rows)
    write_jsonl(eval_path, eval_rows)

    write_cache(
        cache_path,
        {
            "fingerprint": dataset_fingerprint,
            "stats": {
                "raw_rows": len(raw_rows),
                "dropped_rows": dropped,
                "train_rows": len(train_rows),
                "val_rows": len(val_rows),
                "eval_rows": len(eval_rows),
            },
        },
    )

    log.info(
        "data_prep_complete",
        raw_rows=len(raw_rows),
        dropped_rows=dropped,
        train_rows=len(train_rows),
        val_rows=len(val_rows),
        eval_rows=len(eval_rows),
        fingerprint=dataset_fingerprint["raw_sha256"],
    )


def main():
    parser = argparse.ArgumentParser(description="Prepare train/val/eval datasets.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    prepare_data(Path(args.config), force=args.force)


if __name__ == "__main__":
    main()
