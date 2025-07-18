#!/usr/bin/env bash
set -euo pipefail

# SỐ GPU MUỐN DÙNG
NUM_GPUS=4

# THÔNG SỐ CHUNG
MODEL_BASE"HuggingFaceH4/zephyr-7b-beta"
MODEL_PATH="./checkpoints/vis-zephyr-7b-v1-"
IMAGE_FOLDER="./playground/data/finetune/images/"
QUESTION_FILE="./playground/data/vcr-val.json"
VISUAL_PROMPT_STYLE="vcr_qa"
ANSWERS_DIR="./playground/data/eval"
FULL_ANSWERS_FILE="${ANSWERS_DIR}/answers_full.jsonl"

# TẠO THƯ MỤC KẾT QUẢ
mkdir -p "${ANSWERS_DIR}"

echo "=== Chạy đánh giá VCR trên ${NUM_GPUS} GPU ==="

for (( RANK=0; RANK<NUM_GPUS; RANK++ )); do
  CHUNK_FILE="${ANSWERS_DIR}/answers_chunk${RANK}.jsonl"
  echo "[GPU ${RANK}] Xử lý chunk ${RANK}/${NUM_GPUS} → ${CHUNK_FILE}"
  CUDA_VISIBLE_DEVICES=${RANK} python -u -m vis_zephyr.eval.eval_vqa \
    --model-base "${MODEL_BASE}" \
    --model-path "${MODEL_PATH}" \
    --image-folder "${IMAGE_FOLDER}" \
    --question-file "${QUESTION_FILE}" \
    --answers-file "${CHUNK_FILE}" \
    --conv-mode "zephyr_v1" \
    --num_workers 4 \
    --visual_prompt_style "${VISUAL_PROMPT_STYLE}" \
    --temperature 0 \
    --max_new_tokens 128 \
    --image_aspect_ratio anyres \
    --num-chunks ${NUM_GPUS} \
    --chunk-idx ${RANK} \
    --device cuda:0 &

done

# Đợi tất cả tiến trình hoàn tất
wait

echo "=== Ghép kết quả ==="
# Nối tất cả chunk lại thành file cuối
cat ${ANSWERS_DIR}/answers_chunk*.jsonl > "${FULL_ANSWERS_FILE}"

echo "✅ Hoàn tất! File kết quả đầy đủ: ${FULL_ANSWERS_FILE}"

# #!/bin/bash
# python -m vis_zephyr.eval.eval_vqa \
#     --model-base "HuggingFaceH4/zephyr-7b-beta" \
#     --model-path ./checkpoints/vis-zephyr-7b-v1-pretrain \
#     --image-folder ./playground/data/finetune/images/ \
#     --question-file ./playground/data/vcr-val.json \
#     --answers-file ./playground/data/eval/vcr_qa_answers.jsonl \
#     --conv-mode "zephyr_vcr" \
#     --num_workers 4 \
#     --visual_prompt_style vcr_qa \
#     --temperature 0 \
#     --max_new_tokens 128 \
#     --image_aspect_ratio anyres