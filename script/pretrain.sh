#!/bin/bash
# =================================================================================================
# Script: pretrain.sh
# Description: Stage 1 Pre-training script for Vision-Zephyr.
#              This stage focuses exclusively on training the multimodal projector to align
#              the vision encoder with the frozen language model.
# =================================================================================================

# --- Configuration ---
DS_CONFIG_FILE = ./scripts/zero2.json
LLM_BACKBONE   = "HuggingFaceH4/zephyr-7b-beta"
VISION_ENCODER = "openai/clip-vit-large-patch14-336"
DATA_PATH      = "./playground/data/LLaVA-Pretrain/blip.json"
IMAGE_PATH     = "./playground/data/LLaVA-Pretrain/images"
OUTPUT_DIR     = "./checkpoints/vis_zephyr-v1-7b-pretrain"

# --- DeepSpeed Launch Command ---
deepspeed --include localhost:0,1,2,3 vis_zephyr/train/train_mem.py \
    --deepspeed ${DS_CONFIG_FILE} \
    --model_name_or_path ${LLM_BACKBONE} \
    --version zephyr_v1 \
    --data_path ${DATA_PATH} \
    --image_folder ${IMAGE_PATH} \
    --vision_tower ${VISION_ENCODER} \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
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