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

# DATASET=("musique" "samsum" "wikimqa")
# DATASET=("musique")
# DATASET=("samsum")
# DATASET=("wikimqa")
DATASET=("samsum" "wikimqa")

RATIO=("0.0" "0.1" "0.2" "0.3" "0.4" "0.5" "0.6" "0.7" "0.8" "0.9" "1.0")

# for DATASET_NAME in ${DATASET[@]}; do
#     for recomp_ratio in ${RATIO[@]}; do
#         echo "Testing $DATASET_NAME..."
#         log_file=$OUTPUT_DIR/blend_${DATASET_NAME}_mistral.txt
#         cd $ROOT_DIR && python example/blend_${DATASET_NAME}_mistral.py --cache --recomp-ratio $recomp_ratio > $log_file 2>&1
    
#         ttft=$(grep "Avg TTFT:" $log_file | awk '{print $NF}')
#         f1=$(grep "Avg F1:" $log_file | awk '{print $NF}')
#         table_add_row "$TABLE_NAME" "$DATASET_NAME blend-$recomp_ratio $ttft $f1"
#     done
# done

for DATASET_NAME in ${DATASET[@]}; do
    echo "Testing $DATASET_NAME..."
    log_file=$OUTPUT_DIR/blend_${DATASET_NAME}_mistral.txt
    cd $ROOT_DIR && python example/blend_${DATASET_NAME}_mistral.py > $log_file 2>&1
    
    ttft=$(grep "Avg TTFT:" $log_file | awk '{print $NF}')
    f1=$(grep "Avg F1:" $log_file | awk '{print $NF}')
    table_add_row "$TABLE_NAME" "$DATASET_NAME full_prefill $ttft $f1"
done

echo "All tests done. Outputs are saved in $OUTPUT_DIR"