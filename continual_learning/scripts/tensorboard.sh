#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
PORT="${PORT:-6006}"

exec "$PYTHON" -m tensorboard.main \
  --logdir results/babilong_cl/tensorboard \
  --host 0.0.0.0 \
  --port "$PORT"
