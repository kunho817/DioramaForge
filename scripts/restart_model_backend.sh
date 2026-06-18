#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${DIORAMA_MODEL_BACKEND_HOST:-127.0.0.1}"
PORT="${DIORAMA_MODEL_BACKEND_PORT:-9008}"
SESSION="${DIORAMA_MODEL_BACKEND_SESSION:-diorama-model-backend}"

cd "$ROOT"

pkill -f "[p]ython.*model_backend_app.py" || true
if command -v tmux >/dev/null 2>&1; then
  tmux kill-session -t "$SESSION" 2>/dev/null || true
fi
sleep 1

: > model_backend.log

if command -v tmux >/dev/null 2>&1; then
  tmux new-session -d -s "$SESSION" \
    "cd '$ROOT' && source .venv/bin/activate && source .env.remote && python model_backend_app.py --host '$HOST' --port '$PORT' >> model_backend.log 2>&1"
else
  nohup bash -lc "cd '$ROOT' && source .venv/bin/activate && source .env.remote && python model_backend_app.py --host '$HOST' --port '$PORT'" \
    >> model_backend.log 2>&1 < /dev/null &
fi

for _ in 1 2 3 4 5 6 7 8 9 10; do
  sleep 1
  if curl -sS "http://127.0.0.1:${PORT}/api/remote/health"; then
    exit 0
  fi
done

echo "--- model_backend.log ---"
cat model_backend.log
exit 1
