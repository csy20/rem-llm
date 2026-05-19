"""Tests for remllm/eval/quality.py scoring functions (no Ollama needed)."""

from remllm.eval.quality import (
    detect_language,
    extract_code,
    looks_like_code,
    keyword_overlap_score,
    python_syntax_ok,
    js_like_syntax_ok,
    sql_shape_ok,
    score_response,
)


def test_detect_language_python():
    row = {
        "instruction": "Write a Python function",
        "input": "",
        "output": "def foo():",
    }
    assert detect_language(row) == "python"


def test_detect_language_javascript():
    row = {
        "instruction": "Create a React component",
        "input": "",
        "output": "function Button() {}",
    }
    assert detect_language(row) == "javascript"


def test_detect_language_sql():
    row = {
        "instruction": "Write a select query",
        "input": "",
        "output": "SELECT * FROM users",
    }
    assert detect_language(row) == "sql"


def test_detect_language_go():
    row = {"instruction": "Write a Go function", "input": "", "output": "func Foo() {}"}
    assert detect_language(row) == "go"


def test_detect_language_unknown():
    row = {
        "instruction": "Explain this concept",
        "input": "",
        "output": "Here is my explanation.",
    }
    assert detect_language(row) == "unknown"


def test_extract_code_fenced():
    text = "Here's the code:\n```python\nprint('hello')\n```"
    assert "print('hello')" in extract_code(text)


def test_extract_code_unfenced():
    text = "def foo():\n    return 42"
    assert extract_code(text) == "def foo():\n    return 42"


def test_extract_code_multiple_blocks():
    text = "```python\nx = 1\n```\n\n```python\ny = 2\n```"
    code = extract_code(text)
    assert "x = 1" in code
    assert "y = 2" in code


def test_looks_like_code_def():
    assert looks_like_code("def foo():\n    pass")


def test_looks_like_code_function():
    assert looks_like_code("function bar() {}")


def test_looks_like_code_class():
    assert looks_like_code("class MyClass:")


def test_looks_like_code_no():
    assert not looks_like_code("Hello, how are you today?")


def test_keyword_overlap_full():
    score = keyword_overlap_score("def foo bar baz", "def foo bar baz")
    assert score == 1.0


def test_keyword_overlap_partial():
    score = keyword_overlap_score("def foo bar baz qux", "def foo")
    assert 0.0 < score < 1.0


def test_keyword_overlap_empty():
    score = keyword_overlap_score("a b", "c d")
    assert score == 0.0


def test_python_syntax_ok_valid():
    assert python_syntax_ok("def foo():\n    return 42")


def test_python_syntax_ok_invalid():
    assert not python_syntax_ok("def foo(:\n    return 42")


def test_python_syntax_ok_empty():
    assert not python_syntax_ok("")


def test_js_like_syntax_ok_valid():
    assert js_like_syntax_ok("function foo() { return 42; }")


def test_js_like_syntax_ok_unbalanced():
    assert not js_like_syntax_ok("function foo() { return 42;")


def test_js_like_syntax_ok_empty():
    assert not js_like_syntax_ok("")


def test_sql_shape_ok_select():
    assert sql_shape_ok("SELECT * FROM users WHERE id = 1")


def test_sql_shape_ok_insert():
    assert sql_shape_ok("INSERT INTO users (name) VALUES ('test')")


def test_sql_shape_ok_no():
    assert not sql_shape_ok("just some text")


def test_score_response_python():
    row = {
        "instruction": "Write a Python function",
        "input": "",
        "output": "def foo():\n    return 42",
    }
    response = "```python\ndef foo():\n    return 42\n```"
    metrics = score_response(row, response)
    assert metrics["non_empty"] == 1
    assert metrics["has_code"] == 1
    assert metrics["syntax_ok"] == 1
    assert metrics["quality_score"] > 0.5


def test_score_response_empty():
    row = {"instruction": "do something", "input": "", "output": "expected"}
    response = ""
    metrics = score_response(row, response)
    assert metrics["non_empty"] == 0
    assert metrics["quality_score"] < 0.5
