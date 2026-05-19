"""Abstract evaluator interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


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
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        print(f"Wrote report: {path}")


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, model_name: str, rows: list[dict], **kwargs) -> EvalReport: ...
