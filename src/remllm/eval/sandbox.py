"""Sandboxed code execution using Docker containers.

Provides safe evaluation of arbitrary generated code by running it
in an isolated Docker container with resource limits.
"""

import json
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from remllm.logging import get_logger


@dataclass
class SandboxResult:
    """Result from a sandboxed code execution."""

    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    oom_killed: bool = False
    wall_time_s: float = 0.0
    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.oom_killed

    def to_dict(self) -> dict[str, Any]:
        return {
            "exit_code": self.exit_code,
            "stdout": self.stdout[:2000],
            "stderr": self.stderr[:2000],
            "timed_out": self.timed_out,
            "oom_killed": self.oom_killed,
            "wall_time_s": self.wall_time_s,
            "detail": self.detail,
            "passed": self.passed,
        }


LANGUAGE_IMAGE = {
    "python": "python:3.11-slim",
    "javascript": "node:20-slim",
    "typescript": "node:20-slim",
    "bash": "ubuntu:22.04",
    "sql": "python:3.11-slim",
}


LANGUAGE_EXT = {
    "python": ".py",
    "javascript": ".js",
    "typescript": ".ts",
    "bash": ".sh",
    "sql": ".py",
}


LANGUAGE_CMD = {
    "python": ["python3", "-I"],
    "javascript": ["node", "--no-warnings"],
    "typescript": ["npx", "tsx"],
    "bash": ["bash"],
    "sql": ["python3", "-I"],
}


def check_docker_available() -> bool:
    """Check if Docker is installed and the daemon is running."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def run_in_sandbox(
    code: str,
    language: str = "python",
    timeout: int = 30,
    memory_mb: int = 512,
    network_enabled: bool = False,
    mount_readonly: bool = False,
) -> SandboxResult:
    """Execute code in a Docker sandbox.

    Args:
        code: The source code to execute.
        language: Programming language (python, javascript, typescript, bash, sql).
        timeout: Maximum execution time in seconds.
        memory_mb: Memory limit in megabytes.
        network_enabled: Allow network access inside the container.
        mount_readonly: Mount the code as a read-only file.

    Returns:
        SandboxResult with execution details.
    """
    log = get_logger(operation="sandbox_exec", language=language)

    if not check_docker_available():
        log.warning("docker_unavailable", fallback="subprocess")
        return _run_sandbox_fallback(code, language, timeout)

    image = LANGUAGE_IMAGE.get(language, "python:3.11-slim")
    ext = LANGUAGE_EXT.get(language, ".py")
    cmd = LANGUAGE_CMD.get(language, ["python3", "-I"])

    with tempfile.TemporaryDirectory() as tmpdir:
        code_file = Path(tmpdir) / f"code{ext}"
        code_file.write_text(code)

        timeout_script = Path(tmpdir) / "timeout.sh"
        timeout_script.write_text(
            f"#!/bin/bash\ntimeout {timeout} {' '.join(cmd)} /sandbox/code{ext}\n"
        )
        timeout_script.chmod(0o755)

        docker_args = [
            "docker",
            "run",
            "--rm",
            f"--memory={memory_mb}m",
            f"--memory-swap={memory_mb}m",
            "--cpus=1",
            "--network",
            "none" if not network_enabled else "bridge",
            "--read-only" if mount_readonly else "",
            f"--volume={tmpdir}:/sandbox",
            "--workdir=/sandbox",
            image,
            "bash",
            "/sandbox/timeout.sh",
        ]
        docker_args = [a for a in docker_args if a]

        try:
            result = subprocess.run(
                docker_args,
                capture_output=True,
                text=True,
                timeout=timeout + 10,
            )

            sandbox_result = SandboxResult(
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                timed_out=result.returncode == 124,
                detail="",
            )

            if result.returncode == 137:
                sandbox_result.oom_killed = True
                sandbox_result.detail = "out_of_memory"
            elif result.returncode == 124:
                sandbox_result.detail = "timeout"

            return sandbox_result

        except subprocess.TimeoutExpired:
            return SandboxResult(
                exit_code=-1,
                timed_out=True,
                detail="docker_timeout",
            )


def _run_sandbox_fallback(
    code: str,
    language: str = "python",
    timeout: int = 30,
) -> SandboxResult:
    """Fallback subprocess execution when Docker is unavailable."""
    log = get_logger(operation="sandbox_fallback", language=language)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=LANGUAGE_EXT.get(language, ".py"), delete=False
    ) as tmp:
        tmp.write(code)
        tmp_path = tmp.name

    try:
        if language == "python":
            cmd = ["python3", "-I", tmp_path]
        elif language == "javascript":
            cmd = ["node", "--check", tmp_path]
        elif language == "bash":
            cmd = ["bash", "-n", tmp_path]
        elif language == "sql":
            import sqlite3

            conn = sqlite3.connect(":memory:")
            try:
                for stmt in code.split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        conn.execute(stmt)
                conn.close()
                return SandboxResult(exit_code=0, stdout="", detail="sql_passed")
            except Exception as e:
                return SandboxResult(exit_code=1, stderr=str(e), detail="sql_error")
        else:
            cmd = ["python3", "-I", tmp_path]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return SandboxResult(
            exit_code=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )
    except subprocess.TimeoutExpired:
        return SandboxResult(exit_code=-1, timed_out=True, detail="timeout")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def evaluate_with_sandbox(
    model_name: str,
    task_rows: list[dict[str, Any]],
    generate_fn,
    language: str = "python",
    timeout: int = 30,
    max_samples: int | None = None,
) -> list[SandboxResult]:
    """Evaluate a model's code generation using sandboxed execution.

    Args:
        model_name: Model identifier.
        task_rows: List of tasks with 'instruction' and optional 'test'.
        generate_fn: Function (prompt: str) -> str.
        language: Target language.
        timeout: Per-task timeout in seconds.
        max_samples: Max tasks to evaluate.

    Returns:
        List of SandboxResult per task.
    """
    log = get_logger(operation="sandbox_eval", model=model_name, language=language)

    if max_samples:
        task_rows = task_rows[:max_samples]

    results = []
    for i, task in enumerate(task_rows):
        instruction = task.get("instruction", "")
        input_text = task.get("input", "")
        prompt = f"{instruction}\n\n{input_text}".strip()

        try:
            code = generate_fn(prompt)
        except Exception as e:
            results.append(SandboxResult(exit_code=-1, detail=f"generation_error: {e}"))
            continue

        result = run_in_sandbox(code, language=language, timeout=timeout)
        results.append(result)

    passed = sum(1 for r in results if r.passed)
    log.info(
        "sandbox_eval_complete",
        total=len(results),
        passed=passed,
        pass_rate=round(passed / max(len(results), 1), 4),
    )
    return results
