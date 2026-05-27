"""Deduplication pipeline for training data.

Supports exact dedup (SHA-256) and near-dedup (Jaccard on n-gram tokens).
"""

import hashlib
import json
from pathlib import Path


def _token_ngrams(text: str, n: int = 3) -> set[str]:
    tokens = text.lower().split()
    if len(tokens) < n:
        return {text.lower()}
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _jaccard_sim(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def deduplicate_exact(rows: list[dict]) -> tuple[list[dict], int]:
    seen = set()
    deduped = []
    for row in rows:
        text = json.dumps(row, sort_keys=True, ensure_ascii=False)
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            deduped.append(row)
    dropped = len(rows) - len(deduped)
    return deduped, dropped


def deduplicate_near(
    rows: list[dict],
    threshold: float = 0.85,
    key: str = "output",
) -> tuple[list[dict], int]:
    scored = []
    for row in rows:
        text = row.get(key, "") or row.get("instruction", "")
        scored.append((_token_ngrams(text), row))

    deduped = []
    dropped = 0
    for i, (ngrams_i, row) in enumerate(scored):
        is_dup = False
        for j in range(i):
            ng_j, _ = scored[j]
            if _jaccard_sim(ngrams_i, ng_j) >= threshold:
                is_dup = True
                break
        if is_dup:
            dropped += 1
        else:
            deduped.append(row)

    return deduped, dropped


def deduplicate(
    input_path: Path,
    output_path: Path,
    near_dedup: bool = False,
    threshold: float = 0.85,
) -> dict:
    from remllm.data.loader import load_jsonl, write_jsonl

    rows = load_jsonl(input_path)
    original = len(rows)

    rows, exact_dropped = deduplicate_exact(rows)
    near_dropped = 0
    if near_dedup:
        rows, near_dropped = deduplicate_near(rows, threshold=threshold)

    write_jsonl(output_path, rows)
    stats = {
        "original": original,
        "exact_duplicates_removed": exact_dropped,
        "near_duplicates_removed": near_dropped,
        "remaining": len(rows),
    }
    print(json.dumps(stats, indent=2))
    return stats
