#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-30100}"
BASE_URL="${BASE_URL:-http://127.0.0.1:${PORT}}"
MODEL="${SERVED_MODEL_NAME:-Qwen3.8-Flash-Next}"

curl --fail --silent --show-error "$BASE_URL/health" >/dev/null
response="$(curl --fail --silent --show-error \
  "$BASE_URL/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  --data-binary "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of China? Answer with the city name only.\"}],\"temperature\":0,\"max_tokens\":256}")"
python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); c=d["choices"][0]["message"].get("content") or ""; print(c); raise SystemExit(0 if c.strip()=="Beijing" else 1)' <<<"$response"
echo "SMOKE_OK"
