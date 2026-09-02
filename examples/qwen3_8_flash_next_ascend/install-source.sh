#!/usr/bin/env bash
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_ROOT="${SGLANG_ROOT:-/sgl-workspace/sglang}"
TARGET="$SGLANG_ROOT/python/sglang/srt"
EXPECTED_BASE="600df8dd498955f5aef4417383ea9ae5d2c238f7"

if [[ ! -d "$SGLANG_ROOT/.git" ]]; then
  echo "SGLang checkout not found: $SGLANG_ROOT" >&2
  exit 1
fi

current_base="$(git -C "$SGLANG_ROOT" rev-parse HEAD)"
if [[ "$current_base" != "$EXPECTED_BASE" ]]; then
  echo "Unsupported SGLang base: $current_base (expected $EXPECTED_BASE)" >&2
  exit 1
fi

if git -C "$SGLANG_ROOT" apply --reverse --check \
  "$BUNDLE_DIR/sglang-600df8-registration.patch" 2>/dev/null; then
  echo "Registration patch already applied"
else
  git -C "$SGLANG_ROOT" apply --check "$BUNDLE_DIR/sglang-600df8-registration.patch"
  git -C "$SGLANG_ROOT" apply "$BUNDLE_DIR/sglang-600df8-registration.patch"
fi

install -m 0644 "$BUNDLE_DIR/source/qwen4_config.py" "$TARGET/configs/qwen4.py"
install -m 0644 "$BUNDLE_DIR/source/qwen4_exp.py" "$TARGET/configs/qwen4_exp.py"
install -m 0644 "$BUNDLE_DIR/source/qwen4.py" "$TARGET/models/qwen4.py"
install -m 0644 "$BUNDLE_DIR/source/qwen4_layer.py" "$TARGET/models/qwen4_layer.py"
install -m 0644 "$BUNDLE_DIR/source/qwen4_ops.py" "$TARGET/models/qwen4_ops.py"
install -m 0644 "$BUNDLE_DIR/source/npu_graph_runner.py" "$TARGET/model_executor/npu_graph_runner.py"
python3 -m py_compile \
  "$TARGET/configs/qwen4.py" \
  "$TARGET/configs/qwen4_exp.py" \
  "$TARGET/models/qwen4.py" \
  "$TARGET/models/qwen4_layer.py" \
  "$TARGET/models/qwen4_ops.py" \
  "$TARGET/model_executor/npu_graph_runner.py"
