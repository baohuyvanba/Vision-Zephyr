import argparse
import os
import json
import torch
import pandas as pd
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, load_image_from_base64, get_model_name_from_path

def is_none(value):
    return value is None or (isinstance(value, float) and pd.isna(value)) or (isinstance(value, str) and value.strip().lower() in ['none', 'nan'])

def eval_model(args):
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(args.model_path, args.model_base, model_name)

    # Read questions
    questions = pd.read_table(args.question_file)
    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    ans_file = open(args.answers_file, "w")

    # Use conversation template
    conv = conv_templates[args.conv_mode].copy()

    for _, row in tqdm(questions.iterrows(), total=len(questions)):
        idx = row['index']
        question = row['question']
        hint = row['hint']
        image = load_image_from_base64(row['image'])

        # Add hint if available
        if not is_none(hint):
            question = hint + '\n' + question

        # Add options if available
        options = [opt for opt in ['A', 'B', 'C', 'D'] if not is_none(row.get(opt))]
        for opt in options:
            question += f"\n{opt}. {row[opt]}"

        # Add image tokens
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + question
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + question

        # Build prompt
        conv.messages = []  # reset conversation
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Tokenize
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)

        # Process image
        image_tensor = process_images([image], image_processor, model.config)[0].unsqueeze(0).to(model.device, dtype=torch.float16)

        # Generate
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id
            )
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()

        # Write result
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({
            "question_id": idx,
            "prompt": question,
            "text": outputs,
            "options": options,
            "answer_id": ans_id,
            "model_id": model_name,
            "metadata": {}
        }) + "\n")
        ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model checkpoint or HF repo.")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str, required=True, help="Path to .tsv question file")
    parser.add_argument("--answers-file", type=str, required=True, help="Path to output answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    eval_model(args)
