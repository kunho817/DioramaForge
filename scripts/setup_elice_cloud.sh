#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/DioramaForge}"
PERSIST_DIR="${DIORAMA_PERSIST_DIR:-$HOME/diorama-persist}"
REMOTE_WORKDIR="${DIORAMA_REMOTE_WORKDIR:-/tmp/diorama-runs}"

cd "$PROJECT_DIR"

python3 -m venv .venv
# shellcheck source=/dev/null
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
if [[ "${SKIP_OPTIONAL_MODELS:-0}" != "1" ]]; then
  pip install -r requirements-optional-models.txt
fi

mkdir -p "$PERSIST_DIR/hf-cache" "$PERSIST_DIR/models" "$REMOTE_WORKDIR"

cat > .env.remote <<EOF
export HF_HOME="$PERSIST_DIR/hf-cache"
export HF_HUB_CACHE="$PERSIST_DIR/hf-cache/hub"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export DIORAMA_REMOTE_WORKDIR="$REMOTE_WORKDIR"
EOF

cat <<EOF
Remote backend environment prepared.

Before starting the backend, run:
  source .venv/bin/activate
  source .env.remote
  export HF_TOKEN=<set-this-on-server>
  ./scripts/start_model_backend.sh
EOF
