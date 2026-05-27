"""Agent module — wires the context layer into an interactive coding agent.

Orchestrates: AdaptiveRouter → CodebaseIndexer → StructuredOutput → file ops.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from remllm.context.adaptive import AdaptiveRouter, TaskProfile, ToolNeed, classify_task
from remllm.context.indexer import CodebaseIndexer, index_codebase
from remllm.context.structured import StructuredOutput, build_structured_prompt


@dataclass
class AgentResult:
    raw_response: str
    structured: Optional[StructuredOutput] = None
    profile: Optional[TaskProfile] = None
    file_ops: list[str] = field(default_factory=list)
    codebase_chunks: int = 0

    def to_dict(self) -> dict:
        return {
            "raw_response": self.raw_response[:500],
            "plan": self.structured.plan if self.structured else "",
            "operations": self.file_ops,
            "tool_calls": [
                t.name for t in (self.structured.tool_calls if self.structured else [])
            ],
            "profile": self.profile.to_dict() if self.profile else {},
            "codebase_chunks": self.codebase_chunks,
        }


def run_ollama(model: str, prompt: str, timeout_s: int = 120) -> str:
    result = subprocess.run(
        ["ollama", "run", model, prompt],
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ollama run failed")
    return result.stdout.strip()


def run_agent(
    task: str,
    model: str = "deepseek-coder:1.3b",
    project_dir: Optional[Path] = None,
    index_path: Optional[Path] = None,
    dry_run: bool = True,
    timeout_s: int = 120,
) -> AgentResult:
    profile = classify_task(task)

    codebase_context = ""
    indexer: Optional[CodebaseIndexer] = None
    codebase_chunks = 0

    if profile.needs(ToolNeed.CODEBASE_SEARCH) and project_dir and project_dir.exists():
        indexer = CodebaseIndexer(index_path)
        try:
            indexer._load()
        except Exception:
            pass
        if not indexer.chunks:
            indexer.index(project_dir)
        codebase_context = indexer.build_context_prompt(task)
        codebase_chunks = len(indexer.chunks)

    prompt = build_adaptive_prompt(
        task, codebase_context=codebase_context, profile=profile
    )

    raw = run_ollama(model, prompt, timeout_s=timeout_s)

    structured = StructuredOutput.from_response(raw)

    file_ops: list[str] = []
    if structured and structured.operations:
        root = project_dir or Path.cwd()
        file_ops = structured.execute_operations(root, dry_run=dry_run)

    return AgentResult(
        raw_response=raw,
        structured=structured,
        profile=profile,
        file_ops=file_ops,
        codebase_chunks=codebase_chunks,
    )
