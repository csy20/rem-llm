"""Tests for remllm/eval/executable.py (no Ollama needed for these unit tests)."""

import pytest
from remllm.eval.executable import (
    check_python_exec,
    check_javascript_exec,
    check_sql_exec,
    parse_table_definitions,
    evaluate_row,
)


def test_check_python_exec_valid():
    ok, detail = check_python_exec("print('hello')", timeout_s=10)
    assert ok == 1
    assert detail == "ok"


def test_check_python_exec_syntax_error():
    ok, detail = check_python_exec("def foo(:", timeout_s=10)
    assert ok == 0
    assert "syntax_error" in detail


def test_check_python_exec_runtime_error():
    ok, detail = check_python_exec("raise ValueError('boom')", timeout_s=10)
    assert ok == 0


def test_check_python_exec_empty():
    ok, detail = check_python_exec("", timeout_s=10)
    assert ok == 0
    assert "empty" in detail


def test_check_sql_exec_valid():
    ok, detail = check_sql_exec(
        "SELECT * FROM employees", "table employees(id, name, department)"
    )
    assert ok == 1
    assert detail == "ok"


def test_check_sql_exec_invalid_syntax():
    ok, detail = check_sql_exec(
        "SELECT * FROM nonexistent", "table employees(id, name, department)"
    )
    assert ok == 0
    assert "sql_error" in detail


def test_parse_table_definitions():
    text = "table employees(id, name, department) and table projects(id, title)"
    defs = parse_table_definitions(text)
    assert len(defs) == 2
    assert defs[0] == ("employees", ["id", "name", "department"])
    assert defs[1] == ("projects", ["id", "title"])


def test_evaluate_row_python():
    row = {
        "instruction": "Write Python to print hello",
        "input": "",
        "output": "print('hello')",
    }
    response = "```python\nprint('hello')\n```"
    metrics = evaluate_row(row, response, timeout_s=10)
    assert metrics["language"] == "python"
    assert metrics["executable_checked"] == 1
    assert metrics["executable_ok"] == 1


def test_evaluate_row_unsupported():
    row = {
        "instruction": "Explain something",
        "input": "",
        "output": "An explanation",
    }
    response = "Here is the explanation."
    metrics = evaluate_row(row, response, timeout_s=10)
    assert metrics["executable_checked"] == 0
    assert metrics["executable_ok"] == 0
