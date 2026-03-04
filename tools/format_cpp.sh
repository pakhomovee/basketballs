#!/usr/bin/env bash
set -euo pipefail

# Simple helper to auto-format C/C++ files in this repo using clang-format
# according to the top-level .clang-format file.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v clang-format >/dev/null 2>&1; then
  echo "clang-format is not installed. Install it with your package manager." >&2
  exit 1
fi

if [ "$#" -gt 0 ]; then
  FILES="$@"
else
  FILES=$(git ls-files '*.cc' '*.cpp' '*.cxx' '*.c' '*.h' '*.hpp' '*.hh' '*.hxx')
fi

if [ -z "${FILES:-}" ]; then
  echo "No C/C++ files to format."
  exit 0
fi

echo "Formatting C/C++ files:"
echo "$FILES"
clang-format -i $FILES


