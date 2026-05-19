"""Tests for remllm/config.py"""

import tempfile
from pathlib import Path

import pytest
import yaml

from remllm.config import AppConfig, load_config, load_config_dict


VALID_CONFIG = {
    "model": {"base_model_hf": "test/model", "base_model_ollama": "test:latest"},
}


def test_load_config_valid():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(VALID_CONFIG, f)
        f.flush()
        cfg = load_config(Path(f.name))
        assert cfg.model.base_model_hf == "test/model"
        assert cfg.data.train_split == 0.9
        assert cfg.project.seed == 42
    Path(f.name).unlink()


def test_load_config_dict():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(VALID_CONFIG, f)
        f.flush()
        cfg = load_config_dict(Path(f.name))
        assert cfg["model"]["base_model_hf"] == "test/model"
    Path(f.name).unlink()


def test_app_config_defaults():
    cfg = AppConfig.model_validate(VALID_CONFIG)
    assert cfg.project.name == "rem-llm"
    assert cfg.project.seed == 42
    assert cfg.training.backend == "unsloth"
    assert cfg.training.load_in_4bit is True
    assert cfg.data.train_split == 0.9
    assert cfg.hardware.min_gpu_vram_gb_recommended == 8


def test_config_missing_model():
    with pytest.raises(Exception):
        AppConfig.model_validate({})
