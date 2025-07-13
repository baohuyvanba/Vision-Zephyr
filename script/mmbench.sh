#!/bin/bash

python -m vis_zephyr.eval.eval_mmbench \
    --model-path "HuggingFaceH4/zephyr-7b-beta" \
    --question-file "./playground/data/eval/mmbench/mmbench_dev.tsv" \
    --answers-file "./playground/data/eval/mmbench/answers/mmbench_dev/zephyr-7b-beta.jsonl" \
    --conv-mode "zephyr_v1" \
    --temperature 0 \
