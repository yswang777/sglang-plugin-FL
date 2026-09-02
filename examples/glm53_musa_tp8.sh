#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the GLM-5.3-Flash model directory}"

export SGLANG_MUSA_PAGED_MLA="${SGLANG_MUSA_PAGED_MLA:-1}"
export SGLANG_MUSA_ZERO_ROPE_QV_FALLBACK="${SGLANG_MUSA_ZERO_ROPE_QV_FALLBACK:-1}"
export SGLANG_FL_FLAGOS_BLACKLIST="${SGLANG_FL_FLAGOS_BLACKLIST:-count_nonzero,cumsum,mm,unique,_unique2,unique_dim,unique_consecutive,index_put,index_put_,_index_put_impl_}"

exec python -m sglang.launch_server \
  --model-path "${MODEL_PATH}" \
  --tp-size 8 \
  --host 0.0.0.0 \
  --port "${PORT:-30000}" \
  --cuda-graph-max-bs 16 \
  --cuda-graph-backend-decode full
