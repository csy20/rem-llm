#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BASE_OLLAMA_MODEL="${1:-deepseek-coder:1.3b}"
TRAINED_OLLAMA_MODEL="${2:-rem-coder-trained}"

cd "${ROOT_DIR}"
exec python -m remllm.cli pipeline \
  --base-model "${BASE_OLLAMA_MODEL}" \
  --trained-model "${TRAINED_OLLAMA_MODEL}"
