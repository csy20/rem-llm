"""Abstract evaluator interface."""

import json
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable


@dataclass
class EvalReport:
    model: str
    eval_file: str
    num_examples: int
    rates: dict[str, float] = field(default_factory=dict)
    language_rates: dict[str, dict] = field(default_factory=dict)
    samples: list[dict] = field(default_factory=list)
    aggregate: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "eval_file": self.eval_file,
            "num_examples": self.num_examples,
            "aggregate": self.aggregate,
            "rates": self.rates,
            "language_rates": self.language_rates,
            "samples": self.samples,
        }

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        print(f"Wrote report: {path}")


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, model_name: str, rows: list[dict], **kwargs) -> EvalReport: ...


def evaluate_concurrent(
    model_name: str,
    rows: list[dict],
    row_processor: Callable[[str, dict, int], dict],
    max_workers: int = 3,
    timeout_s: int = 180,
) -> list[dict]:
    if max_workers <= 1:
        return [row_processor(model_name, row, timeout_s) for row in rows]

    results: list[dict | None] = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(row_processor, model_name, row, timeout_s): idx
            for idx, row in enumerate(rows)
        }
        for future in as_completed(future_map):
            idx = future_map[future]
            try:
                results[idx] = future.result()
            except Exception as exc:
                results[idx] = {
                    "instruction": rows[idx].get("instruction", ""),
                    "error": str(exc),
                }

    valid = []
    for r in results:
        if r is not None:
            valid.append(r)
    return valid
