#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a
# shellcheck disable=SC1091
source "$BUNDLE_DIR/qwen38.env"
set +a
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ASCEND_LAUNCH_BLOCKING
export PYTHONPATH="/sgl-workspace/sglang/python:${PYTHONPATH:-}"
export SGLANG_FL_FLAGOS_BLACKLIST="${SGLANG_FL_FLAGOS_BLACKLIST:+${SGLANG_FL_FLAGOS_BLACKLIST},}index_copy,index_copy_"
export SGLANG_QWEN4_GRAPH_CANARY=1
export SGLANG_QWEN4_GRAPH_FIXED_REQ_KEYS=1
export SGLANG_QWEN4_GRAPH_INPLACE_STATE=1
export SGLANG_QWEN4_GRAPH_MAX_BS=1
export SGLANG_QWEN4_NOPLE=0

MODEL_PATH="${MODEL_PATH:-/models/Qwen3.8-Flash-Next/}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-30100}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.8-Flash-Next}"

python3 -c 'import tbe; print("TBE_IMPORT_OK")'
exec python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --tp-size 8 \
  --dp-size 2 \
  --load-balance-method round_robin \
  --attention-backend ascend \
  --trust-remote-code \
  --reasoning-parser qwen3-thinking \
  --skip-server-warmup \
  --cuda-graph-bs 1 \
  --cuda-graph-max-bs 1 \
  --disable-radix-cache \
  --chunked-prefill-size 8192 \
  --max-prefill-tokens 32768 \
  --prefill-max-requests 1 \
  --max-total-tokens 140000 \
  --max-running-requests 1 \
  --max-mamba-cache-size 1 \
  --watchdog-timeout 900 \
  --mem-fraction-static 0.88
