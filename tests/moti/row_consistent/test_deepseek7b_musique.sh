#!/bin/bash

CUDA_VISIBLE_DEVICES=1

# Test script for DeepSeek 7B with musique dataset
# This script runs row selection analysis on attention matrix data

# Data directory containing attention matrix exports for DeepSeek 7B + musique
DATA_DIR="../../attn_matrix/attn_exports_deepseek7b_musique"

# Output directory for row selection indices
OUTPUT_DIR="./imp_indices_exports_deepseek7b_musique"

# TopK ratio for row selection (default 0.4)
TOPK=0.4

# Run row selection analysis
python ./row_select_consistent_wsh.py --data ${DATA_DIR} --out ${OUTPUT_DIR} --topk ${TOPK}

# Alternative: use row_select_consistent.py instead
# python ./row_select_consistent.py --data ${DATA_DIR} --out ${OUTPUT_DIR} --topk ${TOPK}

