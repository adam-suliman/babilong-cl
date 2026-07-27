#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
GPU_IDS="${GPU_IDS:-0}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
MICROBATCH_SIZE="${MICROBATCH_SIZE:-1}"
CONFIG="${CONFIG:-$ROOT/continual_learning/configs/qa6_0k_ar_analog_v4.json}"
read -r -a GPUS <<< "$GPU_IDS"

exec "$PYTHON" -m continual_learning suite \
  --config "$CONFIG" \
  --gpus "${GPUS[@]}" \
  --jobs-per-gpu "$JOBS_PER_GPU" \
  --microbatch-size "$MICROBATCH_SIZE" \
  "$@"
