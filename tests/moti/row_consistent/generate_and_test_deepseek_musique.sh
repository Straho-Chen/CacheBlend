#!/bin/bash

set -e

# Parse command-line arguments
MODEL_SIZE="${1:-7B}"  # Default to 7B if not provided
GPU_DEVICE="${2:-${CUDA_VISIBLE_DEVICES}}"  # Use argument, or env var, or prompt
TOPK="${3:-0.4}"  # Default topk ratio

# If GPU_DEVICE is still empty, prompt user or use default
if [ -z "$GPU_DEVICE" ]; then
    echo "CUDA_VISIBLE_DEVICES not set. Please provide GPU device:"
    echo "Usage: $0 [MODEL_SIZE] [GPU_DEVICE] [TOPK]"
    echo "  MODEL_SIZE: 7B or 14B (default: 7B)"
    echo "  GPU_DEVICE: GPU device ID (e.g., 0, 1, 2) or set CUDA_VISIBLE_DEVICES env var"
    echo "  TOPK: TopK ratio for row selection (default: 0.4)"
    echo ""
    echo "Example: $0 7B 2 0.4"
    echo "Or: CUDA_VISIBLE_DEVICES=2 $0 14B"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=${GPU_DEVICE}

# Validate model size
if [ "$MODEL_SIZE" != "7B" ] && [ "$MODEL_SIZE" != "14B" ]; then
    echo "Error: MODEL_SIZE must be either '7B' or '14B'"
    exit 1
fi

echo "=========================================="
echo "Configuration:"
echo "  Model Size: ${MODEL_SIZE}"
echo "  GPU Device: ${GPU_DEVICE}"
echo "  TopK Ratio: ${TOPK}"
echo "=========================================="

# Paths (model size in directory names)
ATTN_MATRIX_DIR="../../attn_matrix"
ATTN_EXPORT_DIR="${ATTN_MATRIX_DIR}/attn_exports_deepseek${MODEL_SIZE}_musique"
OUTPUT_DIR="./imp_indices_exports_deepseek${MODEL_SIZE}_musique"

echo ""
echo "=========================================="
echo "Step 1: Generating attention matrices..."
echo "=========================================="

# Check if attention matrices already exist
if [ -d "${ATTN_EXPORT_DIR}" ] && [ "$(ls -A ${ATTN_EXPORT_DIR}/*.pt 2>/dev/null)" ]; then
    echo "Attention matrices already exist in ${ATTN_EXPORT_DIR}"
    echo "Skipping generation. To regenerate, delete the directory first."
else
    echo "Generating attention matrices for DeepSeek ${MODEL_SIZE} with musique dataset..."
    cd ${ATTN_MATRIX_DIR}
    python ./example/blend_musique_deepseek.py \
        --model-size ${MODEL_SIZE} \
        --enable-think \
        --export-dir ./attn_exports_deepseek${MODEL_SIZE}_musique
    cd - > /dev/null
    echo "Attention matrices generated successfully!"
fi

echo ""
echo "=========================================="
echo "Step 2: Running row selection analysis..."
echo "=========================================="

# Run row selection analysis
python ./row_select_consistent_wsh.py --data ${ATTN_EXPORT_DIR} --out ${OUTPUT_DIR} --topk ${TOPK}

echo ""
echo "=========================================="
echo "All done!"
echo "=========================================="
echo "Attention matrices: ${ATTN_EXPORT_DIR}"
echo "Row selection indices: ${OUTPUT_DIR}"

