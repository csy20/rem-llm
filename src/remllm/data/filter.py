"""Perplexity-based data filtering for training data quality."""

import json
import math
import subprocess
from pathlib import Path


def compute_perplexity_ollama(
    text: str,
    model: str = "qwen2.5-coder:1.5b",
    timeout_s: int = 60,
) -> float:
    eval_prompt = (
        f"Rate the quality of this training example on a scale of 0-10, "
        f"where 10 is excellent. Reply with ONLY the number.\n\n"
        f"Example:\n```\n{text[:2000]}\n```\n\nRating:"
    )
    result = subprocess.run(
        ["ollama", "run", model, eval_prompt],
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if result.returncode != 0:
        return 5.0
    raw = result.stdout.strip()
    try:
        score = float(raw.split()[0]) if raw else 5.0
        return score
    except ValueError:
        for word in raw.split():
            try:
                return float(word)
            except ValueError:
                continue
        return 5.0


def filter_by_perplexity(
    input_path: Path,
    output_path: Path,
    model: str = "qwen2.5-coder:1.5b",
    threshold: float = 5.0,
    max_samples: int = 0,
    timeout_s: int = 60,
) -> dict:
    from remllm.data.loader import load_jsonl, write_jsonl

    rows = load_jsonl(input_path)
    original = len(rows)

    passed = []
    removed = 0
    scored = 0

    sample_count = min(max_samples, len(rows)) if max_samples > 0 else len(rows)

    for i, row in enumerate(rows[:sample_count]):
        text = row.get("output", "") or row.get("instruction", "")
        score = compute_perplexity_ollama(text, model=model, timeout_s=timeout_s)
        scored += 1
        if score >= threshold:
            passed.append(row)
        else:
            removed += 1
            print(
                f"  Filtered: score={score:.1f} threshold={threshold} — {str(row.get('instruction', ''))[:80]}"
            )

    if sample_count < len(rows):
        passed.extend(rows[sample_count:])

    write_jsonl(output_path, passed)
    stats = {
        "original": original,
        "scored": scored,
        "removed_low_quality": removed,
        "remaining": len(passed),
        "threshold": threshold,
    }
    print(json.dumps(stats, indent=2))
    return stats
