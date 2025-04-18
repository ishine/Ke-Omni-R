#!/bin/bash

OUT_DIR=exp/qwen_omni_base689_avqa5k_music5k_think50_reward_weight2_1_g4_beta001_test
#MODEL_NP=/mnt/bella/models/Qwen/Qwen2.5-Omni-7B/
#MODEL_NP=/nfs/172.17.1.38/nvme4/zhaoshuaijiang/models/Qwen2.5-Omni-7B/
MODEL_NP=/nfs/172.17.1.38/nvme4/zhaoshuaijiang/codes/r1-aqa/exp/qwen_omni_base679_avqa5k_music5k_think50_reward_weight2_1_g4_beta001/Qwen2.5-Omni/
DATA_FILE=data/avaq_shuf5000.music_shuf5000.jsonl
#DATA_FILE=data/avaq_shuf12k_music_shuf12k.jsonl

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
    --num_generations 4 \
    --use_wandb false || exit 1
