"""Tests for beginner HTML/CSS/terminal evaluator helpers."""

from remllm.eval.beginner_eval import (
    classify_beginner_task,
    css_quality,
    evaluate_beginner_row,
    html_quality,
    terminal_quality,
)


def test_classify_html():
    row = {"instruction": "Create an HTML page", "input": "", "output": ""}
    assert classify_beginner_task(row) == "html"


def test_classify_css():
    row = {"instruction": "Write CSS for card", "input": "", "output": ""}
    assert classify_beginner_task(row) == "css"


def test_classify_terminal():
    row = {"instruction": "Explain terminal command ls", "input": "", "output": ""}
    assert classify_beginner_task(row) == "terminal"


def test_html_quality_ok():
    code = "<html><body><main><section>Hi</section></main></body></html>"
    ok, detail = html_quality(code)
    assert ok == 1
    assert detail == "ok"


def test_css_quality_ok():
    code = ".card { padding: 12px; border-radius: 8px; }"
    ok, detail = css_quality(code)
    assert ok == 1
    assert detail == "ok"


def test_terminal_quality_risky():
    ok, safety, detail = terminal_quality("rm -rf /tmp")
    assert ok == 0 or ok == 1
    assert safety == 0
    assert detail == "risky_command"


def test_evaluate_beginner_row_terminal_safe():
    row = {"instruction": "Give safe terminal commands", "input": "", "output": ""}
    response = "mkdir demo\ncd demo\ntouch index.html\nls -la"
    metrics = evaluate_beginner_row(row, response)
    assert metrics["task_type"] == "terminal"
    assert metrics["safety_ok"] == 1
