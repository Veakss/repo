#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

trap 'kill 0' EXIT

python -m uvicorn sidecar.main:app --reload --port "${SIDECAR_PORT:-4001}" &
python -m uvicorn backend.main:app --reload --port "${BACKEND_PORT:-8010}" &
python -m streamlit run frontend/app.py --server.port "${STREAMLIT_PORT:-8501}" &

wait
