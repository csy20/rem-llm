"""Data loader utilities: JSONL loading, caching, fingerprinting."""

import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_KEYS = ("instruction", "input", "output")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def validate_rows(rows: list[dict]) -> tuple[list[dict], int]:
    cleaned = []
    dropped = 0
    for row in rows:
        if not all(key in row for key in REQUIRED_KEYS):
            dropped += 1
            continue
        instruction = str(row["instruction"]).strip()
        user_input = str(row["input"]).strip()
        output = str(row["output"]).strip()
        if not instruction or not output:
            dropped += 1
            continue
        if len(output) < 8:
            dropped += 1
            continue
        cleaned.append(
            {
                "instruction": instruction,
                "input": user_input,
                "output": output,
            }
        )
    if not cleaned:
        raise ValueError("All rows were dropped by quality checks.")
    return cleaned, dropped


def build_dataset_fingerprint(raw_path: Path, seed: int, train_split: float) -> dict:
    return {
        "raw_file": str(raw_path),
        "raw_sha256": file_sha256(raw_path),
        "seed": seed,
        "train_split": train_split,
        "required_keys": list(REQUIRED_KEYS),
        "schema_version": 1,
    }


def format_training_row(row: dict) -> dict:
    prompt = row["instruction"]
    if row.get("input"):
        prompt += f"\n\nContext:\n{row['input']}"
    return {"text": f"### Instruction:\n{prompt}\n\n### Response:\n{row['output']}"}
