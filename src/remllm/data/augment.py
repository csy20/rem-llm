"""Data augmentation for training examples.

Augments code data via: variable renaming, comment addition/removal,
function reordering, whitespace variation.
"""

import random
import re
from pathlib import Path


_VAR_NAMES = {
    "python": [
        "data",
        "items",
        "result",
        "value",
        "entry",
        "row",
        "record",
        "obj",
        "cfg",
        "info",
    ],
    "typescript": [
        "data",
        "items",
        "result",
        "value",
        "entry",
        "row",
        "record",
        "obj",
        "cfg",
        "info",
    ],
    "javascript": [
        "data",
        "items",
        "result",
        "value",
        "entry",
        "row",
        "record",
        "obj",
        "cfg",
        "info",
    ],
    "generic": [
        "data",
        "items",
        "result",
        "value",
        "entry",
        "row",
        "record",
        "obj",
        "config",
        "info",
    ],
}


def rename_variables(code: str, domain: str = "generic") -> str:
    names = _VAR_NAMES.get(domain, _VAR_NAMES["generic"])
    var_pattern = re.compile(r"\b(x|y|z|foo|bar|baz|tmp|temp|val|buf)\b")
    used = set()

    def replace(match):
        old = match.group(0)
        new = old
        for n in names:
            if n != old and n not in used:
                new = n
                used.add(n)
                break
        return new

    return var_pattern.sub(replace, code)


def add_comments(code: str) -> str:
    lines = code.split("\n")
    augmented = []
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped and not stripped.startswith(("#", "//", "/*", "*", "'''", '"""')):
            if stripped.startswith("def ") or stripped.startswith("class "):
                augmented.append(line)
                name = (
                    stripped.split("(")[0].split()[-1]
                    if "(" in stripped
                    else stripped.split()[-1].rstrip(":")
                )
                augmented.append(f"    # {name} implementation")
            elif any(
                kw in stripped
                for kw in ["return ", "print(", "console.log", "await ", "yield "]
            ):
                augmented.append(f"    # output/return")
                augmented.append(line)
            else:
                augmented.append(line)
        else:
            augmented.append(line)
    return "\n".join(augmented)


def reorder_functions(code: str) -> str:
    parts = re.split(
        r"(\n(def |class |export function |export const |function |func |fn |pub fn ))",
        code,
    )
    if len(parts) < 5:
        return code
    chunks = []
    for i in range(0, len(parts) - 1, 3):
        header = (
            parts[i] + parts[i + 1] + parts[i + 2] if i + 2 < len(parts) else parts[i]
        )
        body_end = code.find("\n\n", i + len(header))
        if body_end == -1:
            chunks.append(header)
        else:
            chunks.append(code[i:body_end])
    if len(chunks) > 2:
        random.shuffle(chunks)
    return "\n\n".join(chunks)


def augment_row(row: dict, seed: int = 42) -> list[dict]:
    random.seed(seed)
    augmented = [row]

    output = row.get("output", "")
    if not output:
        return augmented

    var_renamed = rename_variables(output)
    if var_renamed != output:
        augmented.append({**row, "output": var_renamed})

    commented = add_comments(output)
    if commented != output:
        augmented.append({**row, "output": commented})

    return augmented[:5]


def augment_dataset(
    input_path: Path,
    output_path: Path,
    factor: int = 3,
    seed: int = 42,
) -> dict:
    from remllm.data.loader import load_jsonl, write_jsonl

    rows = load_jsonl(input_path)
    original = len(rows)
    random.seed(seed)

    augmented = []
    for row in rows:
        variants = augment_row(row, seed=seed + original)
        augmented.extend(variants)
        if len(augmented) >= original * factor:
            break

    write_jsonl(output_path, augmented)
    stats = {
        "original": original,
        "augmented": len(augmented),
        "factor": len(augmented) / max(original, 1),
    }
    print(f"Augmented: {original} → {len(augmented)} rows ({stats['factor']:.1f}x)")
    return stats
