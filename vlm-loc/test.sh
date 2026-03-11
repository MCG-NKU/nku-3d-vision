#!/bin/bash

# For Qwen3-VL-8B
NCCL_P2P_DISABLE=1 \
NCCL_IB_DISABLE=1 \
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model /disk/deepdata/localization/persional/kang/Qwen3-VL-8B-Instruct \
    --adapters checkpoints/output/v0-20251101-125202-qwen3_8b/checkpoint-3600 \
    --infer_backend pt \
    --val_dataset dataset_items/CityLoc-K/vlmloc_testing_data.json \
    --max_new_tokens 2048 \
    --system ./system_prompt.txt \