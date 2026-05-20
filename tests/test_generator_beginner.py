"""Tests for beginner template generation support."""

from remllm.data.generator import BEGINNER_TEMPLATES, generate_dataset


def test_beginner_templates_present():
    assert len(BEGINNER_TEMPLATES) >= 5


def test_generate_beginner_dataset():
    rows = generate_dataset(BEGINNER_TEMPLATES, seed=42)
    assert rows
    domains = {row.get("domain") for row in rows}
    assert "html" in domains
    assert "css" in domains
    assert "terminal" in domains
