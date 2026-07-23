#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
PORT="${PORT:-6006}"

ARGS=(
  --logdir results/babilong_cl/tensorboard
  --host 0.0.0.0
)
PORT_PROVIDED=false
for arg in "$@"; do
  case "$arg" in
    --port|--port=*)
      PORT_PROVIDED=true
      ;;
  esac
done
if [[ "$PORT_PROVIDED" == false ]]; then
  ARGS+=(--port "$PORT")
fi

exec "$PYTHON" -m tensorboard.main "${ARGS[@]}" "$@"
