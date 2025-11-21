#!/bin/bash

set -ex

source "../tools/common.sh"

ABS_PATH=$(where_is_script "$0")

OUTPUT_DIR=$ABS_PATH/output

TABLE_NAME="$ABS_PATH/performance-comparison-table"

table_create "$TABLE_NAME" "model dataset name ttft f1"

mkdir -p $OUTPUT_DIR

# DATASET=("musique" "wikimqa" "samsum")
DATASET=("musique")
# DATASET=("samsum" "wikimqa")
# DATASET=("samsum")
# DATASET=("wikimqa")

# MODEL_SIZE=("14B" "7B")
MODEL_SIZE=("7B")

export LOG_LEVEL=INFO
export CUDA_VISIBLE_DEVICES=1

for size in ${MODEL_SIZE[@]}; do
    for DATASET_NAME in ${DATASET[@]}; do
        echo "Testing $DATASET_NAME mode size $size..."

        # test for think model

        log_file=$OUTPUT_DIR/blend_${DATASET_NAME}_${size}_deepseek_full_reuse.txt
        python example/blend_${DATASET_NAME}_deepseek.py --model-size $size --enable-think --cache --recomp-ratio 0.0 > $log_file 2>&1
        ttft=$(grep "Avg TTFT:" $log_file | awk '{print $NF}')
        f1=$(grep "Avg F1:" $log_file | awk '{print $NF}')
        table_add_row "$TABLE_NAME" "deepseek-think $DATASET_NAME full_reuse-$size $ttft $f1"

        log_file=$OUTPUT_DIR/blend_${DATASET_NAME}_${size}_deepseek_blend.txt
        python example/blend_${DATASET_NAME}_deepseek.py --model-size $size --enable-think --cache --recomp-ratio 0.2 > $log_file 2>&1
        ttft=$(grep "Avg TTFT:" $log_file | awk '{print $NF}')
        f1=$(grep "Avg F1:" $log_file | awk '{print $NF}')
        table_add_row "$TABLE_NAME" "deepseek-think $DATASET_NAME blend-$size $ttft $f1"

        log_file=$OUTPUT_DIR/blend_${DATASET_NAME}_${size}_deepseek_full_prefill.txt
        python example/blend_${DATASET_NAME}_deepseek.py --model-size $size --enable-think > $log_file 2>&1
        ttft=$(grep "Avg TTFT:" $log_file | awk '{print $NF}')
        f1=$(grep "Avg F1:" $log_file | awk '{print $NF}')
        table_add_row "$TABLE_NAME" "deepseek-think $DATASET_NAME full_prefill-$size $ttft $f1"
    done
done

echo "All tests done. Outputs are saved in $OUTPUT_DIR"
