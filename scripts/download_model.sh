#!/usr/bin/env bash
# Download a Q4_K_M GGUF model from HuggingFace.
#
# Usage:  bash scripts/download_model.sh [model]
#
# Default: Qwen2-0.5B-Instruct (Q4_K_M)  — ~350 MB, fits ~900 MB budget
# Alternative: SmolLM2-360M-Instruct     — ~230 MB, even lighter
#
# RAM note:
#   Q4_K_M quantisation means 4-bit weights with K_M medium-size super-block.
#   Good quality/ size trade-off for CPU inference on i3-8100T.

set -euo pipefail

MODEL="${1:-qwen2-0.5b}"
MODELS_DIR="models"

mkdir -p "$MODELS_DIR"

case "$MODEL" in
  qwen2-0.5b)
    REPO="Qwen/Qwen2-0.5B-Instruct-GGUF"
    FILE="qwen2-0_5b-instruct-q4_k_m.gguf"
    ;;
  smollm2-360m)
    REPO="huggingface-quants/SmolLM2-360M-Instruct-Q4_K_M-GGUF"
    FILE="smollm2-360m-instruct-q4_k_m.gguf"
    ;;
  *)
    echo "Unknown model '$MODEL'. Options: qwen2-0.5b, smollm2-360m"
    exit 1
    ;;
esac

if [ -f "$MODELS_DIR/$FILE" ]; then
  echo "Model already exists at $MODELS_DIR/$FILE"
  exit 0
fi

echo "Downloading $FILE from $REPO ..."
curl -L "https://huggingface.co/$REPO/resolve/main/$FILE" \
  -o "$MODELS_DIR/$FILE"

# Symlink as model.gguf so docker-compose.yml always mounts /models/model.gguf
ln -sf "$FILE" "$MODELS_DIR/model.gguf"

echo "Done: $MODELS_DIR/$FILE ($(du -h "$MODELS_DIR/$FILE" | cut -f1))"
echo "Symlinked as $MODELS_DIR/model.gguf"
