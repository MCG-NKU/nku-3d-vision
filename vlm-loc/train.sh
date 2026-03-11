#!/bin/bash

# For Qwen3-VL-8B
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1

swift sft \
    --system ./system_prompt.txt \
    --model /disk/deepdata/localization/persional/kang/Qwen3-VL-8B-Instruct \
    --logging_steps 10 \
    --attn_impl "flash_attn" \
    --dataset dataset_items/CityLoc-K/vlmloc_training_data.json \
    --load_from_cache_file true \
    --val_dataset dataset_items/CityLoc-K/vlmloc_val_data.json \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --target_modules all-linear \
    --freeze_vit false \
    --freeze_aligner false \
    --gradient_accumulation_steps 8 \
    --eval_steps 500 \
    --save_steps 500 \
    --output_dir output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 1 \
    --save_total_limit 5 \
    --gradient_checkpointing true 

