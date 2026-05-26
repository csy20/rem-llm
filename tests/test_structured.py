"""Tests for structured output module."""

from pathlib import Path
from remllm.context.structured import (
    StructuredOutput,
    FileOperation,
    ToolCall,
    build_structured_prompt,
    STRUCTURED_OUTPUT_PROMPT,
)


def test_structured_output_from_json():
    response = """```json
{
  "plan": "Add auth middleware",
  "operations": [
    {"action": "create", "path": "src/middleware/auth.ts", "content": "... auth code"},
    {"action": "modify", "path": "src/app/layout.tsx", "content": "... layout with auth"}
  ],
  "dependencies": ["npm install jose"],
  "tests": [{"path": "tests/auth.test.ts", "content": "test code"}],
  "risks": ["May break existing middleware chain"],
  "notes": ["Tested with Next.js 14"]
}
```"""
    result = StructuredOutput.from_response(response)
    assert result is not None
    assert "auth middleware" in result.plan
    assert len(result.operations) == 2
    assert result.operations[0].action == "create"
    assert result.operations[0].path == "src/middleware/auth.ts"
    assert len(result.dependencies) == 1
    assert len(result.tests) == 1
    assert len(result.risks) == 1
    assert len(result.notes) == 1


def test_structured_output_raw_json():
    response = '{"plan": "Fix bug", "operations": [{"action": "modify", "path": "src/file.ts", "content": "fixed"}]}'
    result = StructuredOutput.from_response(response)
    assert result is not None
    assert result.plan == "Fix bug"
    assert len(result.operations) == 1


def test_structured_output_no_json():
    response = "Here is some text without JSON"
    result = StructuredOutput.from_response(response)
    assert result is None


def test_file_operation_to_dict():
    op = FileOperation(action="create", path="test.ts", content="code")
    output = StructuredOutput(plan="", operations=[op])
    d = output.to_dict()
    assert d["operations"][0]["action"] == "create"
    assert d["operations"][0]["path"] == "test.ts"
    assert d["operations"][0]["content"] == "code"


def test_execute_operations_dry_run(tmp_path: Path):
    output = StructuredOutput(
        plan="Create files",
        operations=[
            FileOperation(
                action="create", path="src/hello.ts", content="export const x = 1"
            ),
            FileOperation(action="modify", path="src/there.ts", content="updated"),
            FileOperation(action="delete", path="old.ts"),
            FileOperation(action="run", path="npm test"),
        ],
    )
    results = output.execute_operations(tmp_path, dry_run=True)
    assert "CREATE src/hello.ts" in results[0]
    assert "MODIFY src/there.ts" in results[1]
    assert "DELETE old.ts" in results[2]
    assert "RUN npm test" in results[3]
    assert not (tmp_path / "src" / "hello.ts").exists()


def test_execute_operations_real(tmp_path: Path):
    output = StructuredOutput(
        plan="Create a file",
        operations=[
            FileOperation(
                action="create", path="src/hello.ts", content="export const x = 1"
            ),
        ],
    )
    output.execute_operations(tmp_path, dry_run=False)
    assert (tmp_path / "src" / "hello.ts").exists()
    assert (tmp_path / "src" / "hello.ts").read_text() == "export const x = 1"


def test_build_structured_prompt():
    prompt = build_structured_prompt("Create a login page", "Use Next.js App Router")
    assert "Add a login page" not in prompt
    assert "Create a login page" in prompt
    assert "Next.js App Router" in prompt
    assert "operations" in prompt
    assert "action" in prompt
