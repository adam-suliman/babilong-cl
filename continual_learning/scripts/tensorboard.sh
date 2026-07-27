#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
PORT="${PORT:-6006}"
RESULTS_ROOT="${RESULTS_ROOT:-results/babilong_cl_v4_ar}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-$RESULTS_ROOT/tensorboard}"

ARGS=(
  --logdir "$TENSORBOARD_DIR"
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
