#!/bin/bash
python -m vis_zephyr.eval.eval_vqa \
    --model-base "HuggingFaceH4/zephyr-7b-beta" \
    --model-path ./checkpoints/vis-zephyr-7b-v1-pretrain \
    --image-folder ./playground/data/finetune/images/ \
    --question-file ./playground/data/vcr-val.json \
    --answers-file ./playground/data/eval/vcr_qa_answers.jsonl \
    --conv-mode "zephyr_vcr" \
    --num_workers 4 \
    --visual_prompt_style vcr_qa \
    --temperature 0 \
    --max_new_tokens 128 \
    --image_aspect_ratio anyres