#!/usr/bin/env bash
# Purge all pipeline data: SQLite database + every job directory (videos,
# annotations, progress files).
# Usage:
#   ./purge.sh            — asks for confirmation
#   ./purge.sh --yes      — skips confirmation (useful in scripts)
set -euo pipefail

WEB_DIR="$(cd "$(dirname "$0")" && pwd)"
STORAGE_DIR="$WEB_DIR/storage"
DB_PATH="$STORAGE_DIR/db.sqlite3"
JOBS_DIR="$STORAGE_DIR/jobs"

if [[ ! -d "$STORAGE_DIR" ]]; then
  echo "Nothing to purge — storage directory does not exist: $STORAGE_DIR"
  exit 0
fi

# Count what will be removed
job_count=0
[[ -d "$JOBS_DIR" ]] && job_count=$(find "$JOBS_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')

echo "This will delete:"
echo "  • Database : $DB_PATH"
echo "  • Jobs     : $job_count job director$([ "$job_count" = 1 ] && echo y || echo ies) under $JOBS_DIR"

if [[ "${1:-}" != "--yes" ]]; then
  read -r -p "Are you sure? [y/N] " confirm
  [[ "$confirm" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 0; }
fi

rm -f "$DB_PATH"
[[ -d "$JOBS_DIR" ]] && rm -rf "$JOBS_DIR"

echo "Purged."
