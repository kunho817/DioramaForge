#!/usr/bin/env bash
set -euo pipefail

HOST="${DIORAMA_MODEL_BACKEND_HOST:-127.0.0.1}"
PORT="${DIORAMA_MODEL_BACKEND_PORT:-9008}"

if [[ -d ".venv" ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

if [[ -f ".env.remote" ]]; then
  # shellcheck source=/dev/null
  source .env.remote
fi

python model_backend_app.py --host "$HOST" --port "$PORT"
