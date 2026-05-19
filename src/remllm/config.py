"""Configuration models and loading for rem-llm."""

from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    name: str = "rem-llm"
    seed: int = 42


class ModelConfig(BaseModel):
    base_model_hf: str
    base_model_ollama: str
    output_name: str = "rem-coder"


class DataConfig(BaseModel):
    raw_file: str = "data/raw.jsonl"
    train_file: str = "data/train.jsonl"
    val_file: str = "data/val.jsonl"
    eval_file: str = "data/eval.jsonl"
    train_split: float = 0.9
    max_length: int = 2048
    pack_sequences: bool = True


class TrainingConfig(BaseModel):
    backend: str = "unsloth"
    output_dir: str = "models/rem-coder-lora"
    merged_output_dir: str = "models/rem-coder-merged"
    epochs: float = 2.0
    learning_rate: float = 0.00012
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 16
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    load_in_4bit: bool = True


class EvalPathsConfig(BaseModel):
    baseline_report: str = "models/evals/baseline.json"
    post_train_report: str = "models/evals/post_train.json"
    baseline_exec_report: str = "models/evals/baseline_exec.json"
    post_train_exec_report: str = "models/evals/post_train_exec.json"


class HardwareConfig(BaseModel):
    min_gpu_vram_gb_recommended: int = 8


class DomainConfig(BaseModel):
    name: str
    extends: Optional[str] = None
    domains: list[str] = Field(default_factory=list)
    training: Optional[TrainingConfig] = None
    data: Optional[DataConfig] = None


class AppConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    model: ModelConfig
    data: DataConfig = Field(default_factory=DataConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: EvalPathsConfig = Field(default_factory=EvalPathsConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        return cls.model_validate(raw or {})


def load_config(path: Path) -> AppConfig:
    return AppConfig.from_yaml(path)


def load_config_dict(path: Path) -> dict:
    """Load raw YAML without Pydantic validation (legacy compat)."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
