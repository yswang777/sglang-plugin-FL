#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a
# shellcheck disable=SC1091
source "$BUNDLE_DIR/qwen38.env"
set +a

MODEL_PATH="${MODEL_PATH:-/data/models/Qwen3.8-Flash-Next}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.8-Flash-Next}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30015}"

exec python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype bfloat16 \
  --tp-size 8 \
  --ep-size 8 \
  --context-length 34000 \
  --max-total-tokens 8192 \
  --max-running-requests 16 \
  --attention-backend fa3 \
  --sampling-backend pytorch \
  --disable-radix-cache \
  --disable-cuda-graph \
  --weight-loader-prefetch-num-threads 16 \
  --mamba-scheduler-strategy extra_buffer \
  --max-mamba-cache-size 80
