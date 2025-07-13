import argparse
import os
import json
import torch
import pandas as pd
from tqdm import tqdm
import shortuuid

from vis_zephyr.constants import *
from vis_zephyr.conversation import templates, SeparatorStyle
from vis_zephyr.model.builder import load_pretrained_model
from vis_zephyr.model.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from vis_zephyr.utils import disable_torch_init
from vis_zephyr.model.multi_scale_process import process_any_resolution_image

from transformers import TextStreamer
from PIL import Image
from io import BytesIO
import base64

def load_image_from_base64(base64_str):
    return Image.open(BytesIO(base64.b64decode(base64_str))).convert('RGB')

def is_none(value):
    return value is None or (isinstance(value, float) and pd.isna(value)) or (isinstance(value, str) and value.strip().lower() in ['none', 'nan'])

def eval_model(args):
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)

    # Load model
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path = args.model_path,
        model_base = args.model_base,
        model_name = model_name,
        device_map = "auto",
        device     = args.device
    )

    # Read questions
    questions = pd.read_table(args.question_file)
    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    ans_file = open(args.answers_file, "w")

    # Conversation template
    conv_mode = args.conv_mode
    conversation = templates[conv_mode].copy()
    roles = conversation.roles

    # Setup stopping criteria
    stop_str = conversation.separator_01 if conversation.separator_style == SeparatorStyle.ZEPHYR else conversation.separator_02
    keywords = [stop_str]

    # Iterate over questions
    for _, row in tqdm(questions.iterrows(), total=len(questions)):
        idx = row['index']
        question = row['question']
        hint = row.get('hint')

        # Add hint if exists
        if not is_none(hint):
            question = hint + "\n" + question

        # Add options if exist
        options = [opt for opt in ['A', 'B', 'C', 'D'] if not is_none(row.get(opt))]
        for opt in options:
            question += f"\n{opt}. {row[opt]}"

        # Add image tokens
        qs = DEFAULT_IMAGE_TOKEN + "\n" + question


        # Reset conversation
        conversation.messages = []
        conversation.append_message(roles[0], qs)
        conversation.append_message(roles[1], None)
        prompt = conversation.get_prompt()

        # Tokenize prompt
        input_ids = tokenizer_image_token(
            prompt=prompt,
            tokenizer=tokenizer,
            return_tensors='pt'
        )

        if len(input_ids.shape) == 1:
            input_ids = input_ids.unsqueeze(0)

        # Process image
        image = load_image_from_base64(row['image'])

        image_tensors = process_any_resolution_image(
            image=image,
            processor=image_processor,
            grid_pinpoints=model.config.mm_grid_pinpoints
        )

        image_tensors = image_tensors.to(dtype = torch.float16)

        if input_ids is None:
            raise ValueError(f"tokenizer_image_token returned None for prompt: {prompt}")
        print(f"input_ids shape: {input_ids.shape}")



        # Stopping criteria & streamer
        stopping_criteria = KeywordsStoppingCriteria(
            keywords  = keywords,
            tokenizer = tokenizer,
            input_ids = input_ids
        )
        streamer = TextStreamer(
            tokenizer   = tokenizer,
            skip_prompt = True,
            skip_special_tokens = True
        )

        # Generate
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                images=image_tensors,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                streamer=streamer
            )
        # Decode
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)].strip()

        # Write answer
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": question,
            "text": outputs,
            "options": options,
            "answer_id": ans_id,
            "model_id": model_name,
            "metadata": {}
        }, ensure_ascii=False) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model checkpoint or HF repo.")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, required=True, help="Path to .tsv question file")
    parser.add_argument("--answers-file", type=str, required=True, help="Path to output answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="zephyr_v1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--debug", action="store_true", help="Print debug info")
    args = parser.parse_args()
    eval_model(args)
