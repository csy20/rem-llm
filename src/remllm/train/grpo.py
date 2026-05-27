"""GRPO reasoning training for coding LLMs.

Implements Group Relative Policy Optimization (GRPO) using Unsloth.
Transforms any standard model into a reasoning model without reasoning traces.
Uses reward functions instead of chain-of-thought examples.
Based on DeepSeek's GRPO and Unsloth's GRPO implementation.
"""

import argparse
import json
import os
from pathlib import Path


def code_correctness_reward(prompt: str, response: str) -> float:
    code = _extract_code(response)
    if not code:
        return 0.0
    score = 0.0
    if _has_syntax(code):
        score += 0.4
    if _has_imports_or_deps(code):
        score += 0.15
    if _has_error_handling(code):
        score += 0.2
    if _has_returns_or_output(code):
        score += 0.15
    if len(response) < 50:
        score *= 0.5
    return score


def format_reward(prompt: str, response: str) -> float:
    if response.strip().startswith("```") or response.strip().startswith("def "):
        return 0.3
    if "```" in response and "```" in response[response.find("```") + 3 :]:
        return 0.6
    return 0.9


def safety_reward(prompt: str, response: str) -> float:
    dangerous_terms = [
        "rm -rf /",
        "DROP TABLE",
        "; --",
        "exec(",
        "eval(",
        "os.system(",
        "subprocess",
        "wget",
        "curl | sh",
    ]
    for term in dangerous_terms:
        if term in response:
            return -0.5
    return 0.3


def _extract_code(text: str) -> str:
    if "```" in text:
        parts = text.split("```")
        for i in range(1, len(parts), 2):
            if i < len(parts):
                code = parts[i]
                if code.startswith("python\n") or code.startswith("python\r\n"):
                    code = code[code.index("\n") + 1 :]
                return code.strip()
    return text.strip()


def _has_syntax(code: str) -> bool:
    import ast

    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _has_imports_or_deps(code: str) -> bool:
    return any(kw in code for kw in ["import ", "from ", "require(", "include("])


def _has_error_handling(code: str) -> bool:
    return any(
        kw in code for kw in ["try:", "try {", "except", "catch (", "if err", ".catch("]
    )


def _has_returns_or_output(code: str) -> bool:
    return any(
        kw in code for kw in ["return ", "print(", "console.log", "echo ", "System.out"]
    )


DEFAULT_REWARDS = [code_correctness_reward, format_reward, safety_reward]


def train_grpo(config_path: Path) -> None:
    try:
        import torch
        from datasets import Dataset
        from trl import GRPOTrainer, GRPOConfig
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise SystemExit(
            "Missing dependencies for GRPO training. Install with:\n"
            "pip install unsloth transformers datasets trl vllm"
        ) from exc

    import yaml

    root = config_path.parent.parent
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    model_cfg = config["model"]
    train_cfg = config["training"]
    seed = int(config["project"]["seed"])

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["base_model_hf"],
        max_seq_length=8192,
        load_in_4bit=True,
        fast_inference=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=seed,
    )

    from remllm.data.loader import load_jsonl

    train_rows = [
        {"prompt": r["instruction"], "response": r["output"]}
        for r in load_jsonl(root / "data/grpo_train.jsonl")
    ]
    train_ds = Dataset.from_list(train_rows)

    grpo_config = GRPOConfig(
        output_dir=str(
            root / train_cfg.get("grpo_output_dir", "models/rem-coder-grpo")
        ),
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        warmup_ratio=0.1,
        logging_steps=5,
        bf16=torch.cuda.is_available(),
        fp16=not torch.cuda.is_available(),
        optim="adamw_8bit",
        max_prompt_length=2048,
        max_completion_length=1024,
        num_generations=4,
        report_to="none",
        seed=seed,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        args=grpo_config,
        train_dataset=train_ds,
        reward_funcs=DEFAULT_REWARDS,
    )

    trainer.train()
    output_dir = root / train_cfg.get("grpo_output_dir", "models/rem-coder-grpo")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved GRPO adapter to {output_dir}")
