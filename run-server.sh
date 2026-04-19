#!/usr/bin/env bash
set -euo pipefail

BIN="/Users/sachinverma/personal/mlx-swift-lm/.build/release/MLXServer"
MODEL="/Users/sachinverma/personal/models/mlx/Qwen3.6-35B-A3B-UD-MLX-4bit"
PORT="${PORT:-8091}"
KV_SCHEME="${KV_SCHEME:-turbo4v2}"
# Slots can be resized at runtime via:
#   curl -s http://127.0.0.1:${PORT}/admin/slots -d '{"count":N}'
# So this is just the startup default.
SLOTS="${SLOTS:-4}"

exec "$BIN" \
  --model "$MODEL" \
  --port "$PORT" \
  --slots "$SLOTS" \
  --kv "$KV_SCHEME"
