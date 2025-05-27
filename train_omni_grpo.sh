#!/bin/bash

OUT_DIR=exp/model
MODEL_NP=path/of/model/Qwen2.5-Omni/
DATA_FILE=data/avaq_shuf5000.music_shuf5000.jsonl

GPU_NUM=$(nvidia-smi -L | wc -l)
NODE_NUM=1
NODE_RANK=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT=32777
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=${GPU_NUM} \
    --nnodes=${NODE_NUM} \
    --node-rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    src/train.py \
    --config_path conf/ds_zero3.json \
    --model_name_or_path ${MODEL_NP} \
    --out_dir ${OUT_DIR} \
    --data_file ${DATA_FILE} \
    --think True \
    --think_max_len 50 \
    --beta 0.01 \
    --num_generations 8 \
    --use_wandb false || exit 1
