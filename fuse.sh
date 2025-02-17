#!/bin/bash

concept_cfg=$1
save_path=$2

python src/gradient_fusion.py \
    --concept_cfg=$1 \
    --save_path=$2 \
    --pretrained_models="experiments/pretrained_models/chilloutmix" \
    --optimize_textenc_iters=500 \
    --optimize_unet_iters=50
