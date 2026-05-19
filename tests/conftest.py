"""Test fixtures and helpers."""

import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def sample_rows():
    return [
        {
            "instruction": "Write a Python function that returns the second largest number.",
            "input": "Input list contains at least two distinct integers.",
            "output": "def second_largest(nums):\n    return sorted(set(nums))[-2]",
        },
        {
            "instruction": "Write a SQL query to return employee count per department.",
            "input": "table employees(id, name, department)",
            "output": "SELECT department, COUNT(*) FROM employees GROUP BY department;",
        },
        {
            "instruction": "Refactor this JavaScript function to handle empty input.",
            "input": "function avg(arr){ return arr.reduce((a,b)=>a+b,0)/arr.length }",
            "output": "function avg(arr) {\n  if (!arr.length) return 0;\n  return arr.reduce((a,b)=>a+b,0)/arr.length\n}",
        },
    ]


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def temp_jsonl(temp_dir, sample_rows):
    path = temp_dir / "test.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in sample_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


@pytest.fixture
def temp_config(temp_dir):
    path = temp_dir / "config.yaml"
    content = {
        "project": {"name": "test", "seed": 42},
        "model": {
            "base_model_hf": "test/model",
            "base_model_ollama": "test-model:latest",
            "output_name": "test-model",
        },
        "data": {
            "raw_file": "data/test.jsonl",
            "train_file": "data/train.jsonl",
            "val_file": "data/val.jsonl",
            "eval_file": "data/eval.jsonl",
            "train_split": 0.8,
            "max_length": 512,
            "pack_sequences": False,
        },
        "training": {
            "backend": "unsloth",
            "output_dir": "models/test-lora",
            "merged_output_dir": "models/test-merged",
            "epochs": 1,
            "learning_rate": 0.0001,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "load_in_4bit": False,
        },
        "evaluation": {
            "baseline_report": "models/evals/baseline.json",
            "post_train_report": "models/evals/post_train.json",
            "baseline_exec_report": "models/evals/baseline_exec.json",
            "post_train_exec_report": "models/evals/post_train_exec.json",
        },
        "hardware": {"min_gpu_vram_gb_recommended": 8},
    }
    with path.open("w", encoding="utf-8") as f:
        import yaml

        yaml.dump(content, f)
    return path
