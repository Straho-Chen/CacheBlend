#!/bin/bash

# Test script for DeepSeek models with musique dataset
# This script runs row selection analysis on attention matrix data
#
# Usage:
#   $0 [MODEL_SIZE] [GPU_DEVICE] [TOPK]
#   MODEL_SIZE: 7B or 14B (default: 7B)
#   GPU_DEVICE: GPU device ID (default: use CUDA_VISIBLE_DEVICES env var)
#   TOPK: TopK ratio (default: 0.4)
#
# Examples:
#   $0 7B 2 0.4
#   CUDA_VISIBLE_DEVICES=1 $0 14B

MODEL_SIZE="${1:-7B}"
GPU_DEVICE="${2:-${CUDA_VISIBLE_DEVICES}}"
TOPK="${3:-0.4}"

# Set GPU device if provided
if [ -n "$GPU_DEVICE" ]; then
    export CUDA_VISIBLE_DEVICES=${GPU_DEVICE}
fi

# Validate model size
if [ "$MODEL_SIZE" != "7B" ] && [ "$MODEL_SIZE" != "14B" ]; then
    echo "Error: MODEL_SIZE must be either '7B' or '14B'"
    exit 1
fi

echo "Configuration: Model=${MODEL_SIZE}, GPU=${CUDA_VISIBLE_DEVICES:-auto}, TopK=${TOPK}"

# Data directory containing attention matrix exports
DATA_DIR="../../attn_matrix/attn_exports_deepseek${MODEL_SIZE}_musique"

# Output directory for row selection indices
OUTPUT_DIR="./imp_indices_exports_deepseek${MODEL_SIZE}_musique"

# Check if data directory exists
if [ ! -d "${DATA_DIR}" ]; then
    echo "Error: Attention matrix directory not found: ${DATA_DIR}"
    echo "Please generate attention matrices first using generate_and_test_deepseek_musique.sh"
    exit 1
fi

# Run row selection analysis
python ./row_select_consistent_wsh.py --data ${DATA_DIR} --out ${OUTPUT_DIR} --topk ${TOPK}

# Alternative: use row_select_consistent.py instead
# python ./row_select_consistent.py --data ${DATA_DIR} --out ${OUTPUT_DIR} --topk ${TOPK}

