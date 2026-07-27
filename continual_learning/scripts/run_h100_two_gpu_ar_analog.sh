#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PROFILE="${PROFILE:-$ROOT/continual_learning/configs/h100_two_gpu_priority.env}"
CONFIG="${CONFIG:-$ROOT/continual_learning/configs/qa6_0k_ar_analog_v4.json}"
# shellcheck source=/dev/null
source "$PROFILE"

PHASE="${1:-all}"
ALL_CONDITIONS=(base-rmt fastmem0 fastmem gpt2 gpt2-si)
REFERENCE_CONDITIONS=(gpt2 base-rmt fastmem0 fastmem)
read -r -a REPLICATES <<< "$REPLICATE_SEEDS"
read -r -a PRIORITY_ORDERS <<< "$PRIORITY_ORDER_SEEDS"
read -r -a EXTENSION_ORDERS <<< "$EXTENSION_ORDER_SEEDS"

run_suite() {
  GPU_IDS="$GPU_IDS" \
  JOBS_PER_GPU="$JOBS_PER_GPU" \
  MICROBATCH_SIZE="$MICROBATCH_SIZE" \
  CONFIG="$CONFIG" \
    bash continual_learning/scripts/run_suite.sh "$@"
}

run_cl() {
  run_suite \
    --conditions "${ALL_CONDITIONS[@]}" \
    --replicate-seeds "${REPLICATES[@]}" \
    --order-seeds "${PRIORITY_ORDERS[@]}" \
    --si-lambda "$SI_LAMBDA" \
    --no-references
}

run_references() {
  run_suite \
    --conditions "${REFERENCE_CONDITIONS[@]}" \
    --replicate-seeds "${REPLICATES[@]}" \
    --order-seeds "$REFERENCE_OWNER_ORDER_SEED" \
    --si-lambda "$SI_LAMBDA"
}

attach_references() {
  run_suite \
    --conditions "${ALL_CONDITIONS[@]}" \
    --replicate-seeds "${REPLICATES[@]}" \
    --order-seeds "${PRIORITY_ORDERS[@]}" \
    --si-lambda "$SI_LAMBDA"
}

run_extension() {
  run_suite \
    --conditions "${ALL_CONDITIONS[@]}" \
    --replicate-seeds "${REPLICATES[@]}" \
    --order-seeds "${EXTENSION_ORDERS[@]}" \
    --si-lambda "$SI_LAMBDA"
}

case "$PHASE" in
  dry-run)
    run_suite \
      --conditions "${ALL_CONDITIONS[@]}" \
      --replicate-seeds "${REPLICATES[@]}" \
      --order-seeds "${PRIORITY_ORDERS[@]}" \
      --si-lambda "$SI_LAMBDA" \
      --no-references \
      --dry-run
    ;;
  cl)
    run_cl
    ;;
  references)
    run_references
    ;;
  attach)
    attach_references
    ;;
  extension)
    run_extension
    ;;
  all)
    run_cl
    run_references
    attach_references
    "$ROOT/.venv/bin/python" -m continual_learning aggregate \
      --results-root results/babilong_cl_v4_ar \
      --output-dir results/babilong_cl_v4_ar/aggregates/latest
    ;;
  *)
    echo "usage: $0 {dry-run|cl|references|attach|extension|all}" >&2
    exit 2
    ;;
esac
