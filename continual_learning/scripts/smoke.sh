#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"

"$PYTHON" -m compileall babilong continual_learning
"$PYTHON" -m continual_learning verify-provenance
CUDA_VISIBLE_DEVICES="" "$PYTHON" -m continual_learning smoke
