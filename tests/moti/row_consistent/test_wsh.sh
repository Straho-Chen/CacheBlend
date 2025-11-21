#!/bin/bash

export CUDA_VISIBLE_DEVICES=2

CUDA_VISIBLE_DEVICES=2

# python ./read_attention_matrix_topk_indices.py --data ../../attn_matrix/attn_exports/ --topk 704 --layer 0 --out ./row_select_layer0_head0_group0

python ./row_select_consistent_wsh.py --data ../../attn_matrix/attn_exports/ --out ./imp_indices_exports

# python ./row_select_consistent.py --data ../../attn_matrix/attn_exports_mistral/ --out ./imp_indices_exports_mistral