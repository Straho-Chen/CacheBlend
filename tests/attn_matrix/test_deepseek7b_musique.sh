#!/bin/bash

# Script to generate attention matrices for DeepSeek models with musique dataset
#
# Usage:
#   $0 [MODEL_SIZE] [GPU_DEVICE]
#   MODEL_SIZE: 7B or 14B (default: 7B)
#   GPU_DEVICE: GPU device ID (default: use CUDA_VISIBLE_DEVICES env var)
#
# Examples:
#   $0 7B 2
#   CUDA_VISIBLE_DEVICES=1 $0 14B

MODEL_SIZE="${1:-7B}"
GPU_DEVICE="${2:-${CUDA_VISIBLE_DEVICES}}"

# Set GPU device if provided
if [ -n "$GPU_DEVICE" ]; then
    export CUDA_VISIBLE_DEVICES=${GPU_DEVICE}
fi

# Validate model size
if [ "$MODEL_SIZE" != "7B" ] && [ "$MODEL_SIZE" != "14B" ]; then
    echo "Error: MODEL_SIZE must be either '7B' or '14B'"
    exit 1
fi

echo "Configuration: Model=${MODEL_SIZE}, GPU=${CUDA_VISIBLE_DEVICES:-auto}"

# Export directory for attention matrices
EXPORT_DIR="./attn_exports_deepseek${MODEL_SIZE}_musique"

# Run the script to generate attention matrices
python ./example/blend_musique_deepseek.py \
    --model-size ${MODEL_SIZE} \
    --enable-think \
    --export-dir ${EXPORT_DIR}

echo "Attention matrices exported to: ${EXPORT_DIR}"

