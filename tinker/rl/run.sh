#!/usr/bin/env bash
# Launch an autonomous research agent on tinker RL (GRPO).
# Usage: ./run.sh [--agent claude|opencode|codex]
set -euo pipefail
cd "$(dirname "$0")"

AGENT="${1:-claude}"
[[ "${1:-}" == "--agent" ]] && AGENT="${2:-claude}"

python3 -c "import yaml" 2>/dev/null || pip3 install -q pyyaml
chmod +x ../../lab
mkdir -p ../../data

PROMPT="You are an autonomous ML research agent. Read program.md for instructions. Then begin. Do not stop until manually interrupted."

case "$AGENT" in
    claude)   claude --dangerously-skip-permissions -p "$PROMPT" ;;
    opencode) opencode -m "$PROMPT" ;;
    codex)    codex "$PROMPT" ;;
    *)        echo "Unsupported: $AGENT (claude, opencode, codex)"; exit 1 ;;
esac
