#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 <gpt2|gpt2-si|base-rmt|fastmem0|fastmem> <replicate-seed> <order-seed> [extra args...]" >&2
  exit 2
fi

CONDITION="$1"
REPLICATE_SEED="$2"
ORDER_SEED="$3"
shift 3

case "$CONDITION" in
  gpt2)
    MODEL="gpt2"
    METHOD="none"
    ;;
  gpt2-si)
    MODEL="gpt2"
    METHOD="si"
    ;;
  base-rmt)
    MODEL="base_rmt"
    METHOD="none"
    ;;
  fastmem0|fastmem)
    MODEL="$CONDITION"
    METHOD="none"
    ;;
  *)
    echo "unknown condition: $CONDITION" >&2
    exit 2
    ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
DEVICE="${DEVICE:-cuda:0}"

exec "$PYTHON" -m continual_learning run \
  --model "$MODEL" \
  --cl-method "$METHOD" \
  --replicate-seed "$REPLICATE_SEED" \
  --order-seed "$ORDER_SEED" \
  --device "$DEVICE" \
  "$@"
