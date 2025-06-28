from transformers import AutoTokenizer
from vis_zephyr.constants import IGNORE_INDEX
from vis_zephyr.conversation import conv_zephyr_v1
from vis_zephyr.model.mm_utils import tokenizer_image_token
from vis_zephyr.train.train import preprocess_zephyr

# Dummy init
conv = conv_zephyr_v1.copy()
sources = [
    [
        {"from": "human", "value": "<image>\nDescribe the image concisely.\n<image>"},
        {"from": "gpt", "value": "a white and black bottle of mens cologne with a black label on it"}
    ],
    [
        {"from": "human", "value": "<image>\nDescribe the image concisely.\n<image>"},
        {"from": "gpt", "value": "a white and black bottle of mens cologne with a black label on it"}
    ]
]

# Tokenization
tokenizer = AutoTokenizer.from_pretrained("models/zephyr-7b-beta", use_fast=False)

results = preprocess_zephyr(sources, tokenizer, has_image = True)

# Inspect
print(f"Input ids shape: {results['input_ids'].shape}")
print(f"Label sample: {results['input_ids'][0]}")
print(f"Input ids shape: {results['labels'].shape}")
print(f"Label sample: {results['labels'][0]}")
