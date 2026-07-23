#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
DEVICE="${DEVICE:-cuda:0}"

exec "$PYTHON" -m continual_learning gpu-smoke \
  --device "$DEVICE" \
  "$@"
