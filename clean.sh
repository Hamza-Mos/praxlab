#!/usr/bin/env bash
# Reset repo to clean state by removing all generated/scaffolded files.
# Safe to run anytime — only removes gitignored and untracked files.
set -euo pipefail

cd "$(dirname "$0")"

echo "Cleaning generated files..."

# Prime scaffolded files (from `prime lab setup`)
rm -rf prime/.claude prime/.prime prime/.gitignore prime/README.md
rm -rf prime/configs prime/environments prime/pyproject.toml prime/uv.lock prime/.venv

# Pretrain generated files
rm -rf pretrain/.venv pretrain/uv.lock

# Python caches
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -name '*.pyc' -delete 2>/dev/null || true

# Log files
find . -name '*.log' -delete 2>/dev/null || true

echo "Done. Run 'git status' to verify."
