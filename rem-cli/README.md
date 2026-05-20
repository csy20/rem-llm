# rem-cli (Rust)

Beginner-focused coding assistant CLI for HTML, CSS, and safe terminal basics.

## Features

- `rem ask "..."` for coding help
- `rem explain "<command>"` for safe terminal guidance
- `rem patch --file <path> --task "..."` for patch previews
- `rem chat` interactive mode
- Structured JSON model contract for stable parsing
- Built-in command safety filtering

## Requirements

- Rust 1.78+
- Ollama running locally
- A local model such as `rem-coder:latest`

## Build

```bash
cargo build
```

## Quick start

```bash
cargo run -- ask "create a simple html page with a header and footer"
cargo run -- explain "rm -rf build"
cargo run -- patch --file index.html --task "add a navigation bar"
cargo run -- chat

# if your model name is different
cargo run -- --model deepseek-coder:1.3b chat
```

## Config

Copy `.remcli.toml.example` to `.remcli.toml` in project root or create
`~/.config/rem-cli/config.toml`.

Supported keys:

- `model`
- `ollama_url`
- `timeout_s`
- `max_context_bytes`
- `prompts_dir`

## Safety model

- Dangerous command patterns are flagged as blocked in output.
- The CLI does not execute shell commands.
- Destructive commands should be replaced by safe previews.

## 404 troubleshooting

If you see `Ollama request failed: 404`:

- ensure Ollama is running: `ollama list`
- run CLI with explicit model: `cargo run -- --model rem-coder:latest chat`
- if base URL includes `/api`, this CLI now handles it automatically
