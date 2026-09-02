#!/usr/bin/env bash
set -euo pipefail

URL="${URL:-http://127.0.0.1:30015}"
MODEL="${SERVED_MODEL_NAME:-Qwen3.8-Flash-Next}"

curl --fail --silent --show-error "$URL/health" >/dev/null
response="$({ printf '%s' "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"中国的首都是什么？请只回答城市名称。\"}],\"temperature\":0,\"max_tokens\":128}"; } | \
  curl --fail --silent --show-error "$URL/v1/chat/completions" \
    -H 'Content-Type: application/json' --data-binary @-)"
python - "$response" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
choice = payload["choices"][0]
content = choice["message"]["content"].strip()
assert choice["finish_reason"] == "stop", choice
assert content.endswith("北京"), content
print("Qwen3.8-Flash-Next MUSA smoke: pass")
PY
