"""Streaming data pipeline — memory-efficient processing for large datasets."""

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterator

from remllm.logging import get_logger


def stream_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Yield rows from a JSONL file one at a time without loading everything into memory."""
    log = get_logger(operation="stream_jsonl", path=str(path))
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                count += 1
                yield row
            except json.JSONDecodeError as e:
                log.warning("skip_malformed_line", error=str(e))
    log.debug("stream_complete", total_rows=count)


def stream_validate(
    rows: Iterator[dict[str, Any]],
    required_keys: tuple[str, ...] = ("instruction", "input", "output"),
    min_output_length: int = 8,
) -> Iterator[dict[str, Any]]:
    """Filter and validate rows in a streaming fashion."""
    log = get_logger(operation="stream_validate")
    kept = 0
    dropped = 0
    for row in rows:
        if not all(k in row for k in required_keys):
            dropped += 1
            continue
        if not row.get("instruction") or not row.get("output"):
            dropped += 1
            continue
        if len(str(row.get("output", ""))) < min_output_length:
            dropped += 1
            continue
        kept += 1
        yield row
    log.debug("validation_complete", kept=kept, dropped=dropped)


def reservoir_sample(
    rows: Iterator[dict[str, Any]],
    k: int,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Reservoir sampling for streaming k rows without loading entire dataset."""
    rng = random.Random(seed)
    sample: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        if i < k:
            sample.append(row)
        else:
            j = rng.randint(0, i)
            if j < k:
                sample[j] = row
    return sample


def batch_iter(
    rows: Iterator[dict[str, Any]], batch_size: int = 1000
) -> Iterator[list[dict[str, Any]]]:
    """Group rows into batches for efficient processing."""
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def stream_split(
    rows: Iterator[dict[str, Any]],
    train_split: float = 0.9,
    seed: int = 42,
    eval_ratio: float = 0.05,
    max_eval: int = 200,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Stream-based train/val/eval split using hashing for deterministic assignment."""
    rng = random.Random(seed)
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []

    for row in rows:
        r = rng.random()
        if r < train_split:
            train_rows.append(row)
        else:
            val_rows.append(row)

    eval_size = min(max(1, int(len(train_rows) * eval_ratio)), max_eval)
    rng.shuffle(val_rows)
    eval_rows = val_rows[:eval_size]

    return train_rows, val_rows, eval_rows


def stream_deduplicate(
    rows: Iterator[dict[str, Any]],
) -> Iterator[dict[str, Any]]:
    """Streaming exact deduplication using rolling hash set."""
    seen: set[str] = set()
    for row in rows:
        h = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            yield row


def stream_to_jsonl(rows: Iterator[dict[str, Any]], path: Path) -> int:
    """Write rows to JSONL one at a time. Returns count."""
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
            count += 1
    return count


def estimate_row_count(path: Path, sample_lines: int = 1000) -> int:
    """Estimate row count from file size and average line length."""
    if not path.exists():
        return 0
    total_bytes = path.stat().st_size
    if total_bytes == 0:
        return 0
    with path.open("r", encoding="utf-8") as f:
        total_chars = 0
        lines = 0
        for line in f:
            total_chars += len(line)
            lines += 1
            if lines >= sample_lines:
                break
    if lines == 0 or total_chars == 0:
        return 0
    avg_chars = total_chars / lines
    return int(total_bytes / avg_chars)


def try_load_hf_dataset(
    path: Path,
    split_name: str = "train",
    streaming: bool = True,
) -> Any | None:
    """Try loading a dataset via HuggingFace Datasets.

    Falls back gracefully if datasets library is not installed or file format
    is not supported.

    Returns a Dataset object or None on failure.
    """
    try:
        from datasets import Dataset, load_dataset
    except ImportError:
        return None

    try:
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            dataset = load_dataset(
                "json",
                data_files=str(path),
                split=split_name,
                streaming=streaming,
            )
        elif suffix in (".parquet", ".arrow"):
            dataset = load_dataset(
                suffix.strip("."),
                data_files=str(path),
                split=split_name,
                streaming=streaming,
            )
        elif suffix == ".json":
            dataset = Dataset.from_json(str(path))
        else:
            dataset = load_dataset(
                "json",
                data_files=str(path),
                split=split_name,
                streaming=streaming,
            )
        return dataset
    except Exception:
        return None


def convert_to_hf_dataset(
    rows: Iterator[dict[str, Any]],
    features: dict | None = None,
    batch_size: int = 10_000,
) -> Any | None:
    """Convert streaming rows to a HuggingFace Dataset, batching to limit memory."""
    try:
        from datasets import Dataset
    except ImportError:
        return None

    batches = []
    batch = []
    total = 0
    for row in rows:
        batch.append(row)
        total += 1
        if len(batch) >= batch_size:
            batches.append(Dataset.from_list(batch))
            batch = []
    if batch:
        batches.append(Dataset.from_list(batch))

    if not batches:
        return None

    if len(batches) == 1:
        return batches[0]

    from datasets import concatenate_datasets

    result = concatenate_datasets(batches)
    return result
