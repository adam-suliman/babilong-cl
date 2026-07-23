#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
"$PYTHON_BIN" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install \
  torch==2.3.1 \
  --index-url https://download.pytorch.org/whl/cu121
.venv/bin/python -m pip install -r requirements.txt

echo "Environment created at $ROOT/.venv"
