#!/bin/bash

# =======================================================================================
#
#       SCRIPT FOR STAGE 1 PRETRAINING (Vision-Zephyr)
#
# Purpose: Train the Multimodal Projector to connect the Vision Encoder with the LLM.
# In this stage, we will:
#   - LLM Backbone (Zephyr) -> Freeze
#   - Vision Encoder (CLIP) -> Freeze
#   - Multimodal Projector (MLP) -> Train
# =======================================================================================
# deepspeed --include localhost:0,1,2,3 vis_zephyr/train/train_mem.py \
deepspeed vis_zephyr/train/train_mem.py \
    --deepspeed ./script/zero2.json \
    --tune_mm_mlp_adapter True \
    --mm_projector_lr 2e-3 \
    --model_name_or_path "HuggingFaceH4/zephyr-7b-beta" \
    --version zephyr_v1 \
    --data_path ./playground/data/pretrain/blip.json \
    --image_folder ./playground/data/pretrain/images/ \
    --vision_tower "openai/clip-vit-large-patch14-336" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer="-2,-5,-8,-11,6" \
    --mm_grid_pinpoints "'[[336, 672], [672, 336], [336, 1008], [1008, 336]]'" \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True\
    --image_aspect_ratio anyres \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/vis-zephyr-7b-v1-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb