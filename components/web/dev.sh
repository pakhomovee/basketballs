#!/usr/bin/env bash
# Run backend + frontend locally without Docker.
# Usage: ./dev.sh [--port 5173] [--port-backend 8000] [--port-frontend 5173] [--view-only]
#   --port is shorthand for --port-frontend (last of --port / --port-frontend wins).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WEB_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_PORT=8000
FRONTEND_PORT=5173
VIEW_ONLY=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --port)
      FRONTEND_PORT="$2"
      shift 2
      ;;
    --port-backend) BACKEND_PORT="$2"; shift 2 ;;
    --port-frontend) FRONTEND_PORT="$2"; shift 2 ;;
    --view-only) VIEW_ONLY=true; shift ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

PIDS=()

cleanup() {
  echo ""
  echo "Shutting down…"
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------
echo "▶ Starting backend on http://localhost:${BACKEND_PORT}"

PYTHONPATH="${REPO_ROOT}/components:${WEB_DIR}" \
  uvicorn web.backend.app:app \
    --host 0.0.0.0 \
    --port "${BACKEND_PORT}" \
    --reload \
    --reload-dir "${WEB_DIR}/backend" \
    --reload-dir "${WEB_DIR}" \
  &
PIDS+=($!)

# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
echo "▶ Starting frontend on http://localhost:${FRONTEND_PORT}"

(
  cd "${WEB_DIR}/frontend"
  if [[ ! -d node_modules ]]; then
    echo "Installing dependencies"
    npm install
  fi
  VITE_VIEW_ONLY="${VIEW_ONLY}" npm run dev -- --port "${FRONTEND_PORT}"
) &
PIDS+=($!)

echo ""
echo "  Backend : http://localhost:${BACKEND_PORT}"
echo "  Frontend: http://localhost:${FRONTEND_PORT}"
echo ""
echo "Press Ctrl-C to stop both."

wait
