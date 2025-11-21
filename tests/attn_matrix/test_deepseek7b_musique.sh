#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

# Export directory for attention matrices
EXPORT_DIR="./attn_exports_deepseek7b_musique"

# Run the script to generate attention matrices
python ./example/blend_musique_deepseek.py \
    --model-size 7B \
    --enable-think \
    --export-dir ${EXPORT_DIR}

echo "Attention matrices exported to: ${EXPORT_DIR}"

