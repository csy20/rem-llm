"""DeepSpeed configuration generators for multi-GPU training."""

import json
from pathlib import Path
from typing import Any

from remllm.logging import get_logger


def generate_ze2_config(
    output_dir: Path,
    batch_size: int = 2,
    grad_accum: int = 16,
    offload_optimizer: bool = False,
    offload_param: bool = False,
) -> dict[str, Any]:
    """Generate DeepSpeed ZeRO-2 configuration.

    ZeRO-2 shards optimizer states and gradients across GPUs.
    Suitable for training 7B models on 4-8 GPUs with 24-80GB VRAM.

    Args:
        output_dir: Directory for checkpoint storage.
        batch_size: Per-device micro batch size.
        grad_accum: Gradient accumulation steps.
        offload_optimizer: Offload optimizer states to CPU (for memory-constrained setups).
        offload_param: Offload model parameters to CPU (extreme memory saving).

    Returns:
        DeepSpeed config dict ready to pass to TrainingArguments.
    """
    return {
        "train_batch_size": batch_size * grad_accum,
        "gradient_accumulation_steps": grad_accum,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": "auto",
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": "auto",
                "warmup_min_lr": 0,
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto",
            },
        },
        "bf16": {"enabled": "auto"},
        "fp16": {"enabled": "auto", "loss_scale": 0, "loss_scale_window": 1000},
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
            "offload_optimizer": {
                "device": "cpu" if offload_optimizer else "none",
            },
            "offload_param": {
                "device": "cpu" if offload_param else "none",
            },
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 10,
    }


def generate_ze3_config(
    output_dir: Path,
    batch_size: int = 2,
    grad_accum: int = 16,
    offload_optimizer: bool = True,
    offload_param: bool = True,
) -> dict[str, Any]:
    """Generate DeepSpeed ZeRO-3 configuration.

    ZeRO-3 shards optimizer, gradients, AND parameters across GPUs.
    Suitable for training 34B+ models on limited GPU memory.

    Args:
        output_dir: Directory for checkpoint storage.
        batch_size: Per-device micro batch size.
        grad_accum: Gradient accumulation steps.
        offload_optimizer: Offload optimizer states to CPU.
        offload_param: Offload model parameters to CPU.
    """
    return {
        "train_batch_size": batch_size * grad_accum,
        "gradient_accumulation_steps": grad_accum,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": "auto",
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": "auto",
                "warmup_min_lr": 0,
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto",
            },
        },
        "bf16": {"enabled": "auto"},
        "fp16": {"enabled": "auto", "loss_scale": 0, "loss_scale_window": 1000},
        "zero_optimization": {
            "stage": 3,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
            "stage3_prefetch_bucket_size": 2e8,
            "stage3_param_persistence_threshold": 1e6,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "offload_optimizer": {
                "device": "cpu" if offload_optimizer else "none",
            },
            "offload_param": {
                "device": "cpu" if offload_param else "none",
            },
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 10,
    }


def write_deepspeed_config(
    config: dict[str, Any],
    path: Path,
) -> Path:
    """Write a DeepSpeed config dict to a JSON file.

    Args:
        config: DeepSpeed configuration dict.
        path: Output file path.

    Returns:
        Path to the written config file.
    """
    log = get_logger(operation="write_deepspeed_config")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(config, f, indent=2)
    log.info("deepspeed_config_written", path=str(path))
    return path


def get_deepspeed_config(
    stage: int = 2,
    output_dir: Path | str = "models/deepspeed",
    batch_size: int = 2,
    grad_accum: int = 16,
    offload_optimizer: bool = False,
    offload_param: bool = False,
) -> dict[str, Any]:
    """Get a DeepSpeed config for the specified stage.

    Args:
        stage: ZeRO stage (2 or 3).
        output_dir: Checkpoint output directory.
        batch_size: Per-device micro batch size.
        grad_accum: Gradient accumulation steps.
        offload_optimizer: CPU offload optimizer states.
        offload_param: CPU offload model parameters.

    Returns:
        DeepSpeed configuration dict.
    """
    output_dir = Path(output_dir)
    if stage == 3:
        return generate_ze3_config(
            output_dir,
            batch_size=batch_size,
            grad_accum=grad_accum,
            offload_optimizer=offload_optimizer,
            offload_param=offload_param,
        )
    return generate_ze2_config(
        output_dir,
        batch_size=batch_size,
        grad_accum=grad_accum,
        offload_optimizer=offload_optimizer,
        offload_param=offload_param,
    )
