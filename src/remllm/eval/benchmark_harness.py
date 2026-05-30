"""Standard benchmark harness — HumanEval, MBPP, MultiPL-E integration.

Evaluates a model against standard coding benchmarks and produces
comparable scores (pass@k, strict accuracy, etc.).
"""

import ast
import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from remllm.logging import get_logger

BENCHMARK_PROMPTS = {
    "humaneval_python": (
        "Complete the following Python function. Return ONLY the function body, "
        "no explanation, no markdown fences.\n\n{prompt}"
    ),
    "mbpp_python": (
        "Write a Python function that solves the following problem. "
        "Return ONLY the function code, no explanation.\n\n{prompt}"
    ),
    "multipl_js": (
        "Complete the following JavaScript function. "
        "Return ONLY the function body, no explanation, no markdown fences.\n\n{prompt}"
    ),
    "multipl_ts": (
        "Complete the following TypeScript function. "
        "Return ONLY the function body, no explanation.\n\n{prompt}"
    ),
    "code_generation": (
        "Generate {language} code for the following task. "
        "Return ONLY the code, no explanation, no markdown fences.\n\n{prompt}"
    ),
}


@dataclass
class BenchmarkResult:
    """Result of running a model against a benchmark."""

    benchmark_name: str
    model_name: str
    total: int = 0
    passed: int = 0
    errors: int = 0
    pass_at_1: float = 0.0
    details: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark_name,
            "model": self.model_name,
            "total": self.total,
            "passed": self.passed,
            "errors": self.errors,
            "pass_at_1": round(self.pass_at_1, 4),
        }

    def to_report(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark_name,
            "model": self.model_name,
            "total": self.total,
            "passed": self.passed,
            "errors": self.errors,
            "pass_at_1": round(self.pass_at_1, 4),
            "details": self.details,
        }


def load_humaneval(path: Path | str) -> list[dict[str, Any]]:
    """Load HumanEval-format JSONL (task_id, prompt, canonical_solution, entry_point, test)."""
    path = Path(path)
    tasks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
    return tasks


def load_mbpp(path: Path | str) -> list[dict[str, Any]]:
    """Load MBPP-format JSONL (task_id, text, code, test_list, test_setup_code)."""
    path = Path(path)
    tasks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
    return tasks


def check_python_solution(
    completion: str,
    test_code: str,
    entry_point: str | None = None,
    timeout: int = 10,
) -> tuple[bool, str]:
    """Check if a Python completion passes the given test code.

    Args:
        completion: The model-generated code.
        test_code: Python test code that asserts correctness.
        entry_point: Function name to test (for HumanEval format).
        timeout: Maximum execution time in seconds.

    Returns:
        (passed, detail) tuple.
    """
    try:
        tree = ast.parse(completion)
    except SyntaxError as e:
        return False, f"syntax_error: {e}"

    full_code = completion + "\n\n" + test_code
    if entry_point:
        full_code += f"\n\ncheck({entry_point})"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
        tmp.write(full_code)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["python3", "-I", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return True, "passed"
        stderr = result.stderr.strip()
        if stderr:
            return False, stderr.split("\n")[-1][:200]
        return False, f"exit_code={result.returncode}"
    except subprocess.TimeoutExpired:
        return False, "timeout"
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def evaluate_humaneval(
    model_name: str,
    benchmark_path: Path | str,
    generate_fn,
    max_samples: int | None = None,
) -> BenchmarkResult:
    """Evaluate a model on HumanEval.

    Args:
        model_name: Name identifier for the model.
        benchmark_path: Path to HumanEval JSONL file.
        generate_fn: async or sync function (prompt: str) -> str.
        max_samples: Limit number of tasks (for quick testing).
    """
    log = get_logger(benchmark="humaneval", model=model_name)
    tasks = load_humaneval(benchmark_path)
    if max_samples:
        tasks = tasks[:max_samples]

    result = BenchmarkResult(
        benchmark_name="humaneval",
        model_name=model_name,
        total=len(tasks),
    )

    for i, task in enumerate(tasks):
        prompt = BENCHMARK_PROMPTS["humaneval_python"].format(prompt=task["prompt"])
        try:
            completion = generate_fn(prompt)
        except Exception as e:
            result.errors += 1
            result.details.append(
                {
                    "task_id": task["task_id"],
                    "status": "generation_error",
                    "error": str(e),
                }
            )
            continue

        passed, detail = check_python_solution(
            completion, task["test"], entry_point=task.get("entry_point")
        )
        if passed:
            result.passed += 1

        result.details.append(
            {
                "task_id": task["task_id"],
                "status": "pass" if passed else "fail",
                "detail": detail,
            }
        )
        log.info(
            "humaneval_task",
            task_id=task["task_id"],
            status="pass" if passed else "fail",
        )

    result.pass_at_1 = result.passed / max(result.total, 1)
    log.info(
        "humaneval_complete",
        total=result.total,
        passed=result.passed,
        errors=result.errors,
        pass_at_1=result.pass_at_1,
    )
    return result


def evaluate_mbpp(
    model_name: str,
    benchmark_path: Path | str,
    generate_fn,
    max_samples: int | None = None,
) -> BenchmarkResult:
    """Evaluate a model on MBPP (Mostly Basic Python Programming).

    Args:
        model_name: Name identifier for the model.
        benchmark_path: Path to MBPP JSONL file.
        generate_fn: Function (prompt: str) -> str.
        max_samples: Limit number of tasks.
    """
    log = get_logger(benchmark="mbpp", model=model_name)
    tasks = load_mbpp(benchmark_path)
    if max_samples:
        tasks = tasks[:max_samples]

    result = BenchmarkResult(
        benchmark_name="mbpp",
        model_name=model_name,
        total=len(tasks),
    )

    for i, task in enumerate(tasks):
        prompt = BENCHMARK_PROMPTS["mbpp_python"].format(prompt=task["text"])
        try:
            completion = generate_fn(prompt)
        except Exception as e:
            result.errors += 1
            result.details.append(
                {
                    "task_id": task.get("task_id", i),
                    "status": "generation_error",
                    "error": str(e),
                }
            )
            continue

        test_code = "\n".join(task.get("test_list", []))
        if task.get("test_setup_code"):
            test_code = task["test_setup_code"] + "\n" + test_code

        passed, detail = check_python_solution(completion, test_code)
        if passed:
            result.passed += 1

        result.details.append(
            {
                "task_id": task.get("task_id", i),
                "status": "pass" if passed else "fail",
                "detail": detail,
            }
        )

    result.pass_at_1 = result.passed / max(result.total, 1)
    log.info(
        "mbpp_complete",
        total=result.total,
        passed=result.passed,
        errors=result.errors,
        pass_at_1=result.pass_at_1,
    )
    return result


def evaluate_on_benchmark(
    model_name: str,
    benchmark_name: str,
    benchmark_path: Path | str,
    generate_fn,
    max_samples: int | None = None,
) -> BenchmarkResult:
    """Unified entry point for running any supported benchmark.

    Args:
        model_name: Model identifier.
        benchmark_name: One of 'humaneval', 'mbpp'.
        benchmark_path: Path to benchmark file.
        generate_fn: Function (prompt: str) -> str.
        max_samples: Optional max tasks limit.
    """
    if benchmark_name == "humaneval":
        return evaluate_humaneval(model_name, benchmark_path, generate_fn, max_samples)
    elif benchmark_name == "mbpp":
        return evaluate_mbpp(model_name, benchmark_path, generate_fn, max_samples)
    else:
        raise ValueError(
            f"Unknown benchmark: {benchmark_name}. Supported: humaneval, mbpp"
        )
