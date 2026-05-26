"""Structured output and tool use — enables machine-parseable code operations.

Supports multi-file code generation and edit operations with a JSON schema
that can be used for prompt engineering and evaluation.

Integrates with the adaptive router for fast tool-use pre-classification.

Dual-mode architecture:
  - CHAT mode: plain text/markdown responses for conversation/questions
  - CODE mode: structured JSON output for file creation/modification
  The system intelligently selects the right mode based on user intent.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from remllm.context.adaptive import TaskProfile


@dataclass
class FileOperation:
    action: str  # create, modify, delete, run
    path: str
    content: Optional[str] = None
    patch: Optional[str] = None


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class StructuredOutput:
    plan: str
    operations: list[FileOperation] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    tests: list[dict] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        ops = []
        for op in self.operations:
            op_dict = {"action": op.action, "path": op.path}
            if op.content:
                op_dict["content"] = op.content
            if op.patch:
                op_dict["patch"] = op.patch
            ops.append(op_dict)

        result: dict[str, Any] = {
            "plan": self.plan,
            "operations": ops,
        }
        if self.tool_calls:
            result["tool_calls"] = [
                {"name": t.name, "arguments": t.arguments} for t in self.tool_calls
            ]
        if self.dependencies:
            result["dependencies"] = self.dependencies
        if self.tests:
            result["tests"] = self.tests
        if self.risks:
            result["risks"] = self.risks
        if self.notes:
            result["notes"] = self.notes
        return result

    @classmethod
    def from_response(cls, response: str) -> Optional["StructuredOutput"]:
        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            try:
                import re

                match = re.search(r"```(?:json)?\s*\n(.*?)\n```", response, re.DOTALL)
                if match:
                    data = json.loads(match.group(1))
                else:
                    return None
            except (json.JSONDecodeError, AttributeError):
                return None

        ops = []
        for op_data in data.get("operations", []):
            if not isinstance(op_data, dict):
                continue
            if "action" not in op_data or "path" not in op_data:
                continue
            ops.append(
                FileOperation(
                    action=op_data["action"],
                    path=op_data["path"],
                    content=op_data.get("content"),
                    patch=op_data.get("patch"),
                )
            )

        tool_calls = []
        for tc in data.get("tool_calls", []):
            if isinstance(tc, dict) and "name" in tc:
                tool_calls.append(
                    ToolCall(name=tc["name"], arguments=tc.get("arguments", {}))
                )

        return cls(
            plan=data.get("plan", ""),
            operations=ops,
            tool_calls=tool_calls,
            dependencies=data.get("dependencies", []),
            tests=data.get("tests", []),
            risks=data.get("risks", []),
            notes=data.get("notes", []),
        )

    def execute_operations(self, root_dir: Path, dry_run: bool = True) -> list[str]:
        results = []
        root = root_dir.resolve()
        for op in self.operations:
            full_path = root_dir / op.path
            resolved = full_path.resolve()
            if not str(resolved).startswith(str(root) + "/"):
                results.append(f"REJECTED {op.path} (path traversal)")
                continue
            if op.action == "create":
                if not dry_run:
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    full_path.write_text(op.content or "", encoding="utf-8")
                results.append(f"CREATE {op.path}")
            elif op.action == "modify":
                if not dry_run:
                    if resolved.exists():
                        resolved.write_text(op.content or "", encoding="utf-8")
                    elif op.patch:
                        existing = resolved.read_text(encoding="utf-8")
                        resolved.write_text(
                            existing + "\n" + op.patch, encoding="utf-8"
                        )
                    elif op.content:
                        resolved.write_text(op.content, encoding="utf-8")
                results.append(f"MODIFY {op.path}")
            elif op.action == "delete":
                if not dry_run:
                    resolved.unlink(missing_ok=True)
                results.append(f"DELETE {op.path}")
            elif op.action == "run":
                results.append(f"RUN {op.path}")
        return results


PROMPT_CHAT_BASE = """[MODE: CHAT]
You are REM, a helpful coding assistant in conversation mode.
Reply with plain text or markdown. NO code generation, NO file creation, NO JSON output.
Be concise and direct. If the user might want code, ask them first:
"Would you like me to write code for that?"
""".strip()

PROMPT_GENERAL = """
[MODE: CODE — GENERAL]
Respond in JSON:
```json
{
  "plan": "Brief description of the approach",
  "operations": [
    {"action": "create|modify|delete", "path": "relative/path.ts", "content": "..."},
    {"action": "run", "path": "command to execute"}
  ],
  "tool_calls": [
    {"name": "web_search", "arguments": {"query": "..."}},
    {"name": "web_fetch", "arguments": {"url": "..."}}
  ],
  "dependencies": ["npm install package-name"],
  "tests": [
    {"path": "test/path.test.ts", "content": "test code"}
  ],
  "risks": ["Potential issue 1"],
  "notes": ["Additional context"]
}
```

Action types:
- create: Write a new file with the provided content
- modify: Replace file content entirely (include full file)
- delete: Remove a file
- run: Execute a shell command
""".strip()

PROMPT_CREATE_ONLY = """
[MODE: CODE — CREATE]
You need to CREATE new files. Output in JSON:
```json
{
  "plan": "Brief description of what files are being created and why",
  "operations": [
    {"action": "create", "path": "relative/path.ts", "content": "complete file content"}
  ],
  "dependencies": ["npm install pkg"],
  "tests": [{"path": "test.test.ts", "content": "tests"}]
}
```
"""
PROMPT_MODIFY_ONLY = """
[MODE: CODE — MODIFY]
You need to MODIFY existing files. Output full file content in JSON:
```json
{
  "plan": "Brief description of the change and why",
  "operations": [
    {"action": "modify", "path": "relative/path.ts", "content": "full updated file content"}
  ],
  "dependencies": []
}
```
"""

PROMPT_WEB_SEARCH = """
[MODE: WEB SEARCH]
You need external knowledge (documentation, API references, how-to). Start by defining what to search:
```json
{
  "plan": "What I need to find",
  "tool_calls": [
    {"name": "web_search", "arguments": {"query": "specific search query"}},
    {"name": "web_fetch", "arguments": {"url": "https://docs.example.com"}}
  ]
}
```
"""

PROMPT_FAST = """
[MODE: CHAT — FAST]
Answer concisely and directly. No tools or code generation needed.
""".strip()

PROMPT_CODEBASE = """
[MODE: CODEBASE SEARCH]
You need to search the existing codebase before responding. First request a codebase search:
```json
{
  "plan": "What I need to find in the codebase",
  "tool_calls": [
    {"name": "codebase_search", "arguments": {"query": "what to search for", "top_k": 5}}
  ]
}
```
"""

STRUCTURED_OUTPUT_PROMPT = PROMPT_GENERAL


def build_structured_prompt(
    task: str,
    context: str = "",
    profile: "Optional[TaskProfile]" = None,
    codebase_context: str = "",
) -> str:
    from remllm.context.adaptive import build_adaptive_prompt, classify_task

    if profile is None:
        profile = classify_task(task)
    return build_adaptive_prompt(task, context, codebase_context, profile)
