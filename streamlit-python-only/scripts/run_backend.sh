#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

python -m uvicorn backend.main:app --reload --port "${BACKEND_PORT:-8010}"
