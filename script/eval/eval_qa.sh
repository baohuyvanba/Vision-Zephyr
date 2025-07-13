#!/bin/bash

OUTPUT_DIR="eval_output"
mkdir -p $OUTPUT_DIR

python -m viszephyr.eval.eval_vqa \
    --model-path ./checkpoints/vis-zephyr-7b-v1-pretrain \
    --image-folder .playground/data/finetune/images/vcr1images/ \
    --question-file $QUESTION_FILE \
    --answers-file $OUTPUT_DIR/vcr_qa_answers.jsonl \
    --visual-prompt-style vcr_qa \
    --temperature 0 \
    --max-new-tokens 128 \
    --image-aspect-ratio anyres