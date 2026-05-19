"""Security evaluator — scans generated code for common vulnerabilities."""

import re

from remllm.eval.base import Evaluator, EvalReport
from remllm.eval.quality import extract_code, run_prompt_ollama


SQL_INJECTION_PATTERNS = [
    (
        r"(?:execute|exec|query)\s*\(\s*['\"]\s*(?:SELECT|INSERT|UPDATE|DELETE).*%(?:s|d)",
        "raw_sql_placeholder",
    ),
    (r"cursor\.execute\s*\(\s*f['\"]", "raw_sql_fstring"),
    (r"\.(?:query|raw)\s*\(\s*`.*\$\{", "prisma_raw_with_interpolation"),
]

XSS_PATTERNS = [
    (
        r"dangerouslySetInnerHTML\s*=\s*\{\s*__html\s*:\s*[^}]+[^(\}]*input",
        "xss_dangerous_html_user_input",
    ),
    (r"document\.write\s*\(\s*", "xss_document_write"),
    (r"eval\s*\(\s*", "xss_eval"),
    (
        r"innerHTML\s*=\s*.*(?:req\.|request\.|params\.|query\.|body\.|input)",
        "xss_innerhtml_user_input",
    ),
]

AUTH_PATTERNS = [
    (
        r"(?:password|secret|token|api_key|apiKey)\s*=\s*['\"][^'\"]{4,}['\"]",
        "hardcoded_secret",
    ),
    (
        r"console\.(?:log|warn|error)\s*\(.*(?:password|token|secret|credential)",
        "leaked_secret_logging",
    ),
    (r"\.use\s*\(\s*(?:app|router)\s*(?!.*auth)", "missing_auth_middleware"),
]

INSECURE_CONFIG_PATTERNS = [
    (r"strict-mode\s*false", "no_strict_mode"),
    (r"http:\/\/", "http_not_https"),
    (r"cors\s*\(\s*\{\s*origin\s*:\s*['\"]\*['\"]\s*\}", "cors_allow_all"),
]


def scan_security(code_text: str) -> dict:
    results = {
        "sql_injection": [],
        "xss": [],
        "auth_secrets": [],
        "insecure_config": [],
    }

    for pattern, label in SQL_INJECTION_PATTERNS:
        if re.search(pattern, code_text, re.IGNORECASE):
            results["sql_injection"].append(label)

    for pattern, label in XSS_PATTERNS:
        if re.search(pattern, code_text, re.IGNORECASE):
            results["xss"].append(label)

    for pattern, label in AUTH_PATTERNS:
        if re.search(pattern, code_text, re.IGNORECASE):
            results["auth_secrets"].append(label)

    for pattern, label in INSECURE_CONFIG_PATTERNS:
        if re.search(pattern, code_text, re.IGNORECASE):
            results["insecure_config"].append(label)

    return results


def check_input_validation(code_text: str) -> dict:
    has_validation = bool(
        re.search(
            r"(?:zod|z\.(?:object|string|number|boolean|array|enum)|joi|yup|express-validator|class-validator|validate|parse)",
            code_text,
            re.IGNORECASE,
        )
    )
    has_sanitization = bool(
        re.search(
            r"(?:sanitize|escape|encodeURI|\.html\s*\()", code_text, re.IGNORECASE
        )
    )
    return {"has_validation": has_validation, "has_sanitization": has_sanitization}


class SecurityEvaluator(Evaluator):
    def evaluate(
        self, model_name: str, rows: list[dict], timeout_s: int = 30, **kwargs
    ) -> EvalReport:
        samples = []
        total_issues = 0
        clean_count = 0

        for row in rows:
            prompt = row["instruction"]
            if row.get("input"):
                prompt = f"{prompt}\n\nContext:\n{row['input']}"

            response = run_prompt_ollama(model_name, prompt, timeout_s=timeout_s)
            code_text = extract_code(response)

            if not code_text.strip():
                samples.append(
                    {"instruction": row["instruction"], "scanned": False, "issues": {}}
                )
                continue

            security_issues = scan_security(code_text)
            validation = check_input_validation(code_text)

            total_issue_count = sum(len(v) for v in security_issues.values())
            total_issues += total_issue_count
            if total_issue_count == 0:
                clean_count += 1

            samples.append(
                {
                    "instruction": row["instruction"],
                    "response_excerpt": response[:300],
                    "security_issues": security_issues,
                    "validation": validation,
                    "clean": total_issue_count == 0,
                }
            )

        return EvalReport(
            model=model_name,
            eval_file="",
            num_examples=len(rows),
            aggregate={"total_issues": total_issues, "clean_samples": clean_count},
            rates={
                "security_clean_rate": round(clean_count / len(rows), 4)
                if rows
                else 0.0,
                "avg_issues_per_sample": round(total_issues / len(rows), 2)
                if rows
                else 0.0,
            },
            samples=samples,
        )
