#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

python ./example/blend_samsum_deepseek.py \
    --model-size 14B \
    --enable-think

# python ./example/blend_samsum_mistral.py --recomp-ratio 0.2 --cache
# python ./example/blend_samsum_mistral.py
