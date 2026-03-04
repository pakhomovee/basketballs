#!/usr/bin/env bash
set -euo pipefail

# Simple helper to auto-format Python files in this repo.
# Uses ruff format + ruff check --fix, configured via pyproject.toml.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v ruff >/dev/null 2>&1; then
  echo "ruff is not installed. Install it with: pip install ruff" >&2
  exit 1
fi

if [ "$#" -gt 0 ]; then
  TARGETS="$@"
else
  TARGETS="."
fi

echo "Running ruff format on: $TARGETS"
ruff format $TARGETS

echo "Running ruff check --fix on: $TARGETS"
ruff check --fix $TARGETS


