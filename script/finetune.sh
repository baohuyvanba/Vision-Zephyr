#!/bin/bash

# ==================================================================================================
#
#       SCRIPT FOR STAGE 2 FINETUNING (Vision-Zephyr with LoRA)
#
# Purpose: Fine-tune the Vision-Zephyr model on a specific multimodal instruction dataset using LoRA.
# In this stage, we will:
#   - LLM Backbone (Zephyr)        -> Train with LoRA (Low-Rank Adaptation)
#   - Vision Encoder (CLIP)        -> Freeze
#   - Multimodal Projector (MLP)   -> Freeze (weights are loaded from pre-training)
#
# ==================================================================================================

deepspeed vis_zephyr/train/train_mem.py \
    --deepspeed ./script/zero2.json \
    --lora_enable True --lora_r 128 --lora_alpha 256 --lora_dropout 0.05 --lora_bias "none" \
    --model_name_or_path "HuggingFaceH4/zephyr-7b-beta" \
    --version "zephyr_v1" \
    --data_path ./playground/data/pretrain/train_gating_subset_10k.json \
    --image_folder ./playground/data/pretrain/images/train_images_10k/ \
    --mm_vision_tower "openai/clip-vit-large-patch14-336" \
    --pretrain_mm_mlp_adapter ./checkpoints/vis-zephyr-7b-v1-pretrain/mm_projector.bin \
    --mm_projector_type "mlp2x_gelu" \
    --mm_vision_select_layer="-2,-5,-8,-11,6" \
    --image_aspect_ratio "pad" \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token True \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/vis-zephyr-7b-v1-finetune-lora \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to "wandb"