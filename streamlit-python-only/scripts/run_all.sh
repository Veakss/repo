#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi
if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  source "$ROOT_DIR/.env"
  set +a
fi

trap 'kill 0' EXIT

"$PYTHON_BIN" -m uvicorn sidecar.main:app --reload --port "${SIDECAR_PORT:-4001}" &
"$PYTHON_BIN" -m uvicorn backend.main:app --reload --port "${BACKEND_PORT:-8010}" &
"$PYTHON_BIN" -m streamlit run frontend/app.py --server.port "${STREAMLIT_PORT:-8501}" &

wait
