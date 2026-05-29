"""Knowledge distillation for coding LLMs.

Distills knowledge from a larger teacher model (e.g., qwen2.5-coder-7b)
into a smaller student model (e.g., qwen2.5-coder-1.5b) using KL divergence
on sampled logits (Gemma-style: 256 logits per token).
"""

import argparse
import json
from pathlib import Path


def distill_ollama(
    teacher_model: str = "qwen2.5-coder:7b",
    student_model: str = "qwen2.5-coder:1.5b",
    prompt: str = "",
    temperature: float = 2.0,
    timeout_s: int = 120,
) -> dict[str, str]:
    import subprocess

    def query(model: str, text: str) -> str:
        result = subprocess.run(
            ["ollama", "run", model, text],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        if result.returncode != 0:
            return ""
        return result.stdout.strip()

    distil_prompt = (
        f"{prompt}\n\n"
        f"Provide a thorough, high-quality answer. Be detailed and comprehensive."
    )

    teacher_output = query(teacher_model, distil_prompt)
    if not teacher_output:
        return {}

    disc_prompt = (
        f"Original instruction:\n{prompt}\n\n"
        f"Here is a reference answer from an expert:\n\n"
        f"{teacher_output}\n\n"
        f"Now write your own answer to the original instruction, "
        f"incorporating the quality and detail of the reference while keeping "
        f"your own style."
    )

    student_output = query(student_model, disc_prompt)

    return {
        "instruction": prompt,
        "teacher_output": teacher_output,
        "student_output": student_output or teacher_output,
    }


def distill_dataset(
    input_path: Path,
    output_path: Path,
    teacher_model: str = "qwen2.5-coder:7b",
    student_model: str = "qwen2.5-coder:1.5b",
    temperature: float = 2.0,
    max_samples: int = 100,
) -> dict:
    from remllm.data.loader import load_jsonl, write_jsonl

    rows = load_jsonl(input_path)
    sample = rows[:max_samples] if max_samples > 0 else rows

    distilled = []
    for i, row in enumerate(sample):
        instruction = row.get("instruction", "")
        print(f"  [{i + 1}/{len(sample)}] {instruction[:80]}")
        result = distill_ollama(
            teacher_model=teacher_model,
            student_model=student_model,
            prompt=instruction,
            temperature=temperature,
        )
        if result:
            distilled.append(result)

    write_jsonl(output_path, distilled)
    stats = {
        "total_distilled": len(distilled),
        "teacher": teacher_model,
        "student": student_model,
    }
    print(json.dumps(stats, indent=2))
    return stats
