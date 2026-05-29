#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# ── REM CLI Installer ──────────────────────────────────────────────────────
# Usage: curl -fsSL https://raw.githubusercontent.com/<user>/rem-llm/main/install.sh | bash
#
# Detects OS and architecture, downloads the matching binary from GitHub Releases,
# installs to ~/.local/bin/, and adds it to PATH if needed.

REPO="csy20/rem-llm"
VERSION="${VERSION:-latest}"
INSTALL_DIR="${HOME}/.local/bin"
BINARY="rem"
BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
DIM="\033[2m"
RED="\033[31m"
RESET="\033[0m"

info()  { echo -e "  ${GREEN}✓${RESET} $*"; }
warn()  { echo -e "  ${YELLOW}!${RESET} $*"; }
header() { echo -e "\n${BOLD}┃ REM Installer${RESET} ${DIM}───────────────────────────${RESET}\n"; }
step()  { echo -e "  ${DIM}│${RESET} $*"; }

abort() {
    echo -e "  ${RED}✗${RESET} Error: $*" >&2
    exit 1
}

header

# ── Detect platform ──────────────────────────────────────────────────────────
step "detecting platform..."

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"

case "$OS" in
    linux)  PLATFORM_OS="linux" ;;
    darwin) PLATFORM_OS="macos" ;;
    *)      abort "unsupported OS: $OS (only Linux and macOS are supported)" ;;
esac

case "$ARCH" in
    x86_64|amd64)   PLATFORM_ARCH="x86_64" ;;
    aarch64|arm64)  PLATFORM_ARCH="aarch64" ;;
    *)              abort "unsupported architecture: $ARCH" ;;
esac

TARGET="${PLATFORM_ARCH}-${PLATFORM_OS}"
step "platform: ${BOLD}${TARGET}${RESET}"

# ── Get latest version if not specified ───────────────────────────────────────
if [ "$VERSION" = "latest" ]; then
    step "fetching latest release..."
    VERSION=$(curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
        | grep '"tag_name"' | head -1 | sed 's/.*"tag_name": "\(.*\)".*/\1/')
    if [ -z "$VERSION" ]; then
        abort "could not determine latest version"
    fi
fi
step "version: ${BOLD}${VERSION}${RESET}"

# ── Download binary ──────────────────────────────────────────────────────────
BINARY_NAME="rem-${TARGET}"
if [ "$PLATFORM_OS" = "macos" ]; then
    BINARY_NAME="${BINARY_NAME}.tar.gz"
fi

DOWNLOAD_URL="https://github.com/${REPO}/releases/download/${VERSION}/${BINARY_NAME}"
step "downloading ${BINARY_NAME}..."
step "  ${DIM}from: ${DOWNLOAD_URL}${RESET}"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

cd "$TMP_DIR"

if ! curl -fSL --progress-bar "$DOWNLOAD_URL" -o "$BINARY_NAME"; then
    abort "failed to download binary. Check that the release exists at: $DOWNLOAD_URL"
fi

# Extract if tar.gz
if [[ "$BINARY_NAME" == *.tar.gz ]]; then
    tar xzf "$BINARY_NAME"
fi

chmod +x rem 2>/dev/null || true

# ── Install ──────────────────────────────────────────────────────────────────
mkdir -p "$INSTALL_DIR"
cp rem "$INSTALL_DIR/${BINARY}" 2>/dev/null || {
    mv rem "$INSTALL_DIR/${BINARY}" 2>/dev/null || abort "failed to install binary"
}
info "installed to ${BOLD}${INSTALL_DIR}/${BINARY}${RESET}"

# ── Add to PATH if needed ────────────────────────────────────────────────────
SHELL_NAME="$(basename "${SHELL:-bash}")"
SHELL_RC=""

case "$SHELL_NAME" in
    bash) SHELL_RC="${HOME}/.bashrc" ;;
    zsh)  SHELL_RC="${HOME}/.zshrc" ;;
    fish) SHELL_RC="${HOME}/.config/fish/config.fish" ;;
esac

if [ -n "$SHELL_RC" ] && ! echo "$PATH" | grep -q "$INSTALL_DIR"; then
    if [ "$SHELL_NAME" = "fish" ]; then
        echo "fish_add_path $INSTALL_DIR" >> "$SHELL_RC"
    else
        echo "export PATH=\"\$HOME/.local/bin:\$PATH\"" >> "$SHELL_RC"
    fi
    warn "added ${BOLD}${INSTALL_DIR}${RESET} to ${SHELL_RC}"
    warn "restart your shell or run: ${BOLD}source ${SHELL_RC}${RESET}"
fi

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
info "REM CLI installed successfully!"
echo ""
echo "  Run:  ${BOLD}rem${RESET}       — start interactive chat"
echo "  Run:  ${BOLD}rem ask \"...\"${RESET}  — ask a coding question"
echo "  Run:  ${BOLD}rem new <name>${RESET} — scaffold a project"
echo ""
echo "  ${DIM}Requires Ollama: https://ollama.com${RESET}"
echo "  ${DIM}Recommended: ollama pull qwen2.5-coder:1.5b${RESET}"
echo ""

# ── Ollama environment hints ────────────────────────────────────────────────
if command -v ollama &>/dev/null; then
    warn "For low-RAM machines (4–6GB), set these env vars:"
    echo ""
    echo "  export OLLAMA_FLASH_ATTENTION=1    # 30-50% KV cache RAM savings"
    echo "  export OLLAMA_KV_CACHE_TYPE=q8_0   # half precision KV cache"
    echo "  export OLLAMA_MMAP=1               # mmap model load"
    echo "  export OLLAMA_MAX_LOADED_MODELS=1  # keep one model loaded"
    echo ""
fi
