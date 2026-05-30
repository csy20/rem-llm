"""DPO (Direct Preference Optimization) training pipeline.

Trains a model to prefer chosen (good) responses over rejected (bad) responses
using paired comparison data, without needing a separate reward model.
"""

import os
from pathlib import Path
from typing import Any

from remllm.config import load_config_dict, resolve_project_root
from remllm.data.loader import load_jsonl
from remllm.experiment import ExperimentTracker
from remllm.logging import get_logger


def _format_dpo_row(
    row: dict[str, Any], chat_template: bool = True
) -> dict[str, Any | list]:
    """Format a row into DPO format: prompt, chosen, rejected."""
    instruction = row.get("instruction", "")
    input_text = row.get("input", "")
    chosen = row.get("chosen") or row.get("output", "")
    rejected = row.get("rejected") or row.get("bad_output", "")

    if chat_template:
        prompt = [
            {"role": "system", "content": ""},
            {"role": "user", "content": f"{instruction}\n\n{input_text}".strip()},
        ]
    else:
        prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def train_dpo(config_path: Path | str, run_id: str | None = None) -> None:
    """Run DPO fine-tuning on preference pairs.

    Expects data with 'chosen' and 'rejected' fields alongside the usual
    'instruction'/'input' format. Falls back to using 'output' as chosen
    and 'bad_output' as rejected if dedicated fields are missing.

    Args:
        config_path: Path to config YAML.
        run_id: Optional experiment run ID.
    """
    log = get_logger(phase="dpo_training")
    tracker = ExperimentTracker(
        run_id=run_id, backend=os.environ.get("REMLLM_TRACKER", "local")
    )

    config = load_config_dict(config_path)
    root = resolve_project_root(config_path, str(config["data"]["raw_file"]))
    data_cfg = config["data"]
    train_cfg = config["training"]
    model_cfg = config["model"]

    train_path = root / data_cfg.get("train_file", data_cfg["raw_file"])
    dpo_output_dir = Path(
        train_cfg.get("dpo_output_dir", train_cfg["output_dir"] + "-dpo")
    )

    rows = load_jsonl(train_path)
    dpo_rows = [_format_dpo_row(r, train_cfg.get("chat_template", True)) for r in rows]
    log.info("dpo_data_loaded", total_rows=len(dpo_rows))

    try:
        from trl import DPOTrainer
        from unsloth import FastLanguageModel
    except ImportError as e:
        log.error("import_error", error=str(e))
        log.info("dpo_requires", packages="trl>=0.8, unsloth")
        return

    base_model = model_cfg["base_model_hf"]
    max_length = int(data_cfg.get("max_length", 4096))
    load_in_4bit = train_cfg.get("load_in_4bit", True)
    lora_rank = int(train_cfg.get("lora_rank", 32))
    lora_alpha = int(train_cfg.get("lora_alpha", 64))
    lora_dropout = float(train_cfg.get("lora_dropout", 0.05))
    learning_rate = float(train_cfg.get("learning_rate", 5e-5))
    epochs = int(train_cfg.get("epochs", 2))
    batch_size = int(train_cfg.get("batch_size", 2))
    grad_accum = int(train_cfg.get("grad_accum", 8))
    beta = float(train_cfg.get("dpo_beta", 0.1))

    tracker.log_params(
        {
            "base_model": base_model,
            "max_length": max_length,
            "load_in_4bit": load_in_4bit,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "dpo_beta": beta,
            "train_samples": len(dpo_rows),
        }
    )

    log.info("loading_base_model", model=base_model)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    from transformers import TrainingArguments

    training_args = TrainingArguments(
        output_dir=str(dpo_output_dir),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        report_to=_detect_report_to(),
        run_name=tracker.run_id,
    )

    import torch

    train_dataset = [
        {
            "prompt": r["prompt"],
            "chosen": r["chosen"],
            "rejected": r["rejected"],
        }
        for r in dpo_rows
    ]

    dpo_trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        beta=beta,
        max_length=max_length,
        max_prompt_length=max_length // 2,
    )

    log.info("dpo_training_starting")
    dpo_trainer.train()
    log.info("dpo_training_complete")

    model.save_pretrained(str(dpo_output_dir))
    tokenizer.save_pretrained(str(dpo_output_dir))
    log.info("dpo_adapter_saved", output_dir=str(dpo_output_dir))

    tracker.log_artifact(dpo_output_dir)
    tracker.log_dict(
        {"train_samples": len(dpo_rows), "output_dir": str(dpo_output_dir)},
        "dpo_summary",
    )
    tracker.finish()


def _detect_report_to() -> list[str]:
    """Detect which reporting backends to use."""
    reporters = []
    if os.environ.get("WANDB_PROJECT") or os.environ.get("WANDB_MODE"):
        reporters.append("wandb")
    if os.environ.get("TENSORBOARD_LOG_DIR"):
        reporters.append("tensorboard")
    return reporters or ["none"]
