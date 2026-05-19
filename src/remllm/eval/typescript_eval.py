"""TypeScript evaluator — checks generated TypeScript code for type errors using tsc."""

import subprocess
import tempfile
from pathlib import Path

from remllm.eval.base import Evaluator, EvalReport
from remllm.eval.quality import extract_code, detect_language, run_prompt_ollama


def check_typescript_typecheck(code_text: str, timeout_s: int = 30) -> tuple[int, str]:
    if not code_text.strip():
        return 0, "empty_code"

    with tempfile.TemporaryDirectory(prefix="rem_ts_") as temp_dir:
        ts_file = Path(temp_dir) / "generated.ts"
        tsconfig = Path(temp_dir) / "tsconfig.json"
        ts_file.write_text(code_text, encoding="utf-8")
        tsconfig.write_text(
            """{
  "compilerOptions": {
    "strict": true,
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "jsx": "react-jsx",
    "esModuleInterop": true,
    "skipLibCheck": true,
    "noEmit": true
  }
}""",
            encoding="utf-8",
        )

        process = subprocess.run(
            ["npx", "--yes", "typescript", "--noEmit", "--project", str(temp_dir)],
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
        if process.returncode != 0:
            error_lines = process.stdout.strip().splitlines()
            errors = [l for l in error_lines if "error TS" in l]
            return 0, f"tsc_errors:{len(errors)}"
        return 1, "ok"


def check_import_structure(code_text: str) -> tuple[int, str]:
    issues = []
    lines = code_text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("import ") and "from" not in stripped:
            issues.append(f"L{i + 1}: malformed import")
        if "require(" in stripped:
            issues.append(f"L{i + 1}: CommonJS require detected, prefer ESM imports")
        if stripped.startswith("export default") and not stripped.endswith((";", "}")):
            pass
    if issues:
        return 0, "; ".join(issues[:3])
    return 1, "ok"


def check_nextjs_conventions(code_text: str) -> dict:
    has_export = "export" in code_text
    has_default = "export default" in code_text
    has_server_actions = "'use server'" in code_text
    has_client_directive = "'use client'" in code_text
    has_generate_metadata = (
        "generateMetadata" in code_text or "generateStaticParams" in code_text
    )

    issues = []
    if has_server_actions and "async " not in code_text:
        issues.append("server_action_not_async")
    if has_client_directive and has_server_actions:
        issues.append("mixed_client_server_directives")

    return {
        "has_export": has_export,
        "has_default_export": has_default,
        "has_server_directive": has_server_actions,
        "has_client_directive": has_client_directive,
        "has_next_metadata": has_generate_metadata,
        "issues": issues,
    }


class TypeScriptEvaluator(Evaluator):
    def evaluate(
        self, model_name: str, rows: list[dict], timeout_s: int = 30, **kwargs
    ) -> EvalReport:
        samples = []
        type_ok_count = 0
        checked_count = 0

        for row in rows:
            if detect_language(row) not in ("javascript", "unknown"):
                continue

            prompt = row["instruction"]
            if row.get("input"):
                prompt = f"{prompt}\n\nContext:\n{row['input']}"

            response = run_prompt_ollama(model_name, prompt, timeout_s=timeout_s)
            code_text = extract_code(response)

            if not code_text.strip():
                samples.append(
                    {"instruction": row["instruction"], "type_ok": 0, "detail": "empty"}
                )
                continue

            type_ok, detail = check_typescript_typecheck(code_text, timeout_s)
            import_ok, import_detail = check_import_structure(code_text)
            nextjs_conventions = check_nextjs_conventions(code_text)
            checked_count += 1

            if type_ok:
                type_ok_count += 1

            samples.append(
                {
                    "instruction": row["instruction"],
                    "response_excerpt": response[:300],
                    "type_ok": type_ok,
                    "type_detail": detail,
                    "import_ok": import_ok,
                    "import_detail": import_detail,
                    "nextjs_conventions": nextjs_conventions,
                }
            )

        return EvalReport(
            model=model_name,
            eval_file="",
            num_examples=len(rows),
            aggregate={"checked": checked_count, "type_ok": type_ok_count},
            rates={
                "typescript_type_ok_rate": round(type_ok_count / checked_count, 4)
                if checked_count
                else 0.0,
                "checked_rate": round(checked_count / len(rows), 4) if rows else 0.0,
            },
            samples=samples,
        )
