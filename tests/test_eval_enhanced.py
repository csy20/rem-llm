"""Tests for security and TypeScript evaluators."""

import pytest
from remllm.eval.security_eval import (
    scan_security,
    check_input_validation,
    SQL_INJECTION_PATTERNS,
    XSS_PATTERNS,
    AUTH_PATTERNS,
)


def test_scan_security_clean():
    code = "function hello() { return 'hello world'; }"
    results = scan_security(code)
    for key in results:
        assert results[key] == [], f"Expected no {key} issues"


def test_scan_sql_injection():
    code = "cursor.execute(f'SELECT * FROM users WHERE id = {user_id}')"
    results = scan_security(code)
    assert len(results["sql_injection"]) > 0


def test_scan_xss():
    code = "element.innerHTML = request.body.content"
    results = scan_security(code)
    assert len(results["xss"]) > 0


def test_scan_hardcoded_secret():
    code = "const API_KEY = 'sk-abc123def456'"
    results = scan_security(code)
    assert len(results["auth_secrets"]) > 0


def test_scan_http_url():
    code = "fetch('http://example.com/api')"
    results = scan_security(code)
    assert len(results["insecure_config"]) > 0


def test_check_input_validation_with_zod():
    code = "const schema = z.object({ name: z.string() })"
    result = check_input_validation(code)
    assert result["has_validation"] is True


def test_check_input_validation_without():
    code = "const name = req.body.name"
    result = check_input_validation(code)
    assert result["has_validation"] is False


def test_check_input_validation_with_sanitize():
    code = "const safe = sanitizeHtml(input)"
    result = check_input_validation(code)
    assert result["has_sanitization"] is True
