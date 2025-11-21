#!/bin/bash

set -e

export CUDA_VISIBLE_DEVICES=2
CUDA_VISIBLE_DEVICES=2

# Paths
ATTN_MATRIX_DIR="../../attn_matrix"
ATTN_EXPORT_DIR="${ATTN_MATRIX_DIR}/attn_exports_deepseek7b_musique"
OUTPUT_DIR="./imp_indices_exports_deepseek7b_musique"
TOPK=0.4

echo "=========================================="
echo "Step 1: Generating attention matrices..."
echo "=========================================="

# Check if attention matrices already exist
if [ -d "${ATTN_EXPORT_DIR}" ] && [ "$(ls -A ${ATTN_EXPORT_DIR}/*.pt 2>/dev/null)" ]; then
    echo "Attention matrices already exist in ${ATTN_EXPORT_DIR}"
    echo "Skipping generation. To regenerate, delete the directory first."
else
    echo "Generating attention matrices for DeepSeek 7B with musique dataset..."
    cd ${ATTN_MATRIX_DIR}
    python ./example/blend_musique_deepseek.py \
        --model-size 7B \
        --enable-think \
        --export-dir ./attn_exports_deepseek7b_musique
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

