# REM LLM - Coding Model

Fine-tuned LLM for coding tasks using existing open-source model.

## Quick Start

### 1. Install Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Pull a base coding model
```bash
ollama pull codellama
# or
ollama pull deepseek-coder
```

### 3. Fine-tune options

**Option A - Ollama + Modelfile (No GPU needed, basic)**
- Create a Modelfile with custom prompts
- `ollama create rem-coder -f Modelfile`

**Option B - LlamaFactory (GPU recommended)**
```bash
pip install llamafactory
```
- Supports LoRA, QLoRA fine-tuning
- Requires: 8GB+ VRAM GPU

**Option C - Unsloth (Fastest, most memory efficient)**
```bash
pip install unsloth
```
- 2x faster training, 70% less memory
- Supports: Llama, Mistral, Phi models

## Project Structure
```
rem-llm/
├── data/           # Training data (JSONL format)
├── models/        # Saved model weights
├── scripts/       # Training scripts
└── config/        # Configuration files
```

## Training Data Format
Prepare data in JSONL format:
```json
{"instruction": "Write a function to sort a list", "input": "", "output": "def sort_list(lst): ..."}
```