#!/bin/bash

set -ex

source "../tools/common.sh"

ABS_PATH=$(where_is_script "$0")

ROOT_DIR=$ABS_PATH/../../

OUTPUT_DIR=$ABS_PATH/output

TABLE_NAME="$ABS_PATH/performance-comparison-table"

table_create "$TABLE_NAME" "dataset name ttft f1"

echo "root_dir should be the repo path! current root_dir is $ROOT_DIR"

mkdir -p $OUTPUT_DIR

DATASET=("musique" "samsum" "wikimqa")
# DATASET=("musique")

MODEL_SIZE=("14B" "7B")

for DATASET_NAME in ${DATASET[@]}; do
    for size in ${MODEL_SIZE[@]}; do
        echo "Testing $DATASET_NAME mode size $size..."
        log_file=$OUTPUT_DIR/blend_${DATASET_NAME}_${size}_deepseek.txt
        cd $ROOT_DIR && python example/blend_${DATASET_NAME}_deepseek.py --model-size $size > $log_file 2>&1

        blend_ttft=$(grep "Avg TTFT with cache:" $log_file | awk '{print $NF}')
        blend_f1=$(grep "Avg F1 with cache:" $log_file | awk '{print $NF}')
        table_add_row "$TABLE_NAME" "$DATASET_NAME blend-$size $blend_ttft $blend_f1"

        full_reuse_ttft=$(grep "Avg TTFT with full reuse:" $log_file | awk '{print $NF}')
        full_reuse_f1=$(grep "Avg F1 with full reuse:" $log_file | awk '{print $NF}')
        table_add_row "$TABLE_NAME" "$DATASET_NAME full_reuse-$size $full_reuse_ttft $full_reuse_f1"

        full_prefill_ttft=$(grep "Avg TTFT with full prefill:" $log_file | awk '{print $NF}')
        full_prefill_f1=$(grep "Avg F1 with full prefill:" $log_file | awk '{print $NF}')
        table_add_row "$TABLE_NAME" "$DATASET_NAME full_prefill-$size $full_prefill_ttft $full_prefill_f1"
    done
done

echo "All tests done. Outputs are saved in $OUTPUT_DIR"
