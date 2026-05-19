"""Structured output and tool use — enables machine-parseable code operations.

Supports multi-file code generation and edit operations with a JSON schema
that can be used for prompt engineering and evaluation.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class FileOperation:
    action: str  # create, modify, delete
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
        for op in self.operations:
            full_path = root_dir / op.path
            if op.action == "create":
                if not dry_run:
                    full_path.parent.mkdir(parents=True, exist_ok=True)
                    full_path.write_text(op.content or "", encoding="utf-8")
                results.append(f"CREATE {op.path}")
            elif op.action == "modify":
                if not dry_run:
                    existing = (
                        full_path.read_text(encoding="utf-8")
                        if full_path.exists()
                        else ""
                    )
                    if op.content:
                        full_path.write_text(op.content, encoding="utf-8")
                    elif op.patch:
                        full_path.write_text(
                            existing + "\n" + op.patch, encoding="utf-8"
                        )
                results.append(f"MODIFY {op.path}")
            elif op.action == "delete":
                if not dry_run:
                    full_path.unlink(missing_ok=True)
                results.append(f"DELETE {op.path}")
            elif op.action == "run":
                results.append(f"RUN {op.path}")
        return results


STRUCTURED_OUTPUT_PROMPT = """
When responding with code changes, output in this JSON format:

```json
{
  "plan": "Brief description of the approach",
  "operations": [
    {"action": "create|modify|delete", "path": "relative/path.ts", "content": "..."},
    {"action": "run", "path": "command to execute"}
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


def build_structured_prompt(task: str, context: str = "") -> str:
    prompt = f"{STRUCTURED_OUTPUT_PROMPT}\n\nTask: {task}"
    if context:
        prompt += f"\n\nContext:\n{context}"
    prompt += "\n\nRespond with only the JSON output."
    return prompt
