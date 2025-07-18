import argparse
import torch
import os
import json
from tqdm import tqdm
import math
import copy
import random
import shortuuid
import re

from torch.utils.data import Dataset, DataLoader
from PIL import Image

from vis_zephyr.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from vis_zephyr.model.builder import load_pretrained_model
from vis_zephyr.utils import disable_torch_init
from vis_zephyr.model.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from vis_zephyr.model.vip_processor.processor import visual_prompt_process
from vis_zephyr.conversation import templates, SeparatorStyle
from vis_zephyr.model.multi_scale_process import process_any_resolution_image

def extract_answer(output_string: str) -> str:
    # Bước 1: Tìm các ký tự A, B, C, D viết hoa không phải chữ cái đầu câu
    # Điều kiện: Ký tự phải đứng sau một khoảng trắng, dấu chấm, dấu phẩy, hoặc dấu ngoặc.
    # Hoặc đứng độc lập như "(A)", "A."
    matches = re.findall(r"(?<=[ .,(\[])([ABCD])(?=[ .,)\]])", output_string)
    if matches:
        return matches[0]

    # Bước 2: Tìm ký tự A, B, C, D viết hoa không phải chữ cái đầu tiên của câu (từ câu thứ 2 trở đi)
    # Ví dụ: "Câu 1. Đáp án là B. Câu 2. A là đúng." -> Lấy 'A'
    sentences = re.split(r'(?<=[.!?])\s+', output_string)
    if len(sentences) > 1:
        for sentence in sentences[1:]: # Bắt đầu từ câu thứ 2
            # Tìm ký tự A, B, C, D đứng độc lập (không phải một phần của từ)
            # Ví dụ: " A.", "(B)", " C "
            isolated_matches = re.findall(r"(?<![a-zA-Z0-9])([ABCD])(?![a-zA-Z0-9])", sentence)
            if isolated_matches:
                return isolated_matches[0]

    # Bước 3: Lấy ký tự đầu tiên trong toàn bộ chuỗi nếu là A, B, C, D
    first_char_match = re.match(r"^[ABCD]", output_string.strip())
    if first_char_match:
        return first_char_match.group(0)

    # Bước cuối cùng: Nếu không tìm thấy, trả về "A"
    return "A"

#Divide into chunks for distributed evaluation
def split_list(list, n):
    """Split a list into n chunks (nearly equal parts)"""
    chunk_size = math.ceil(len(list) / n)
    return [list[i:i + chunk_size] for i in range(0, len(list), chunk_size)]

def get_chunk(list, n, k):
    chunks = split_list(list, n)
    return chunks[k]

#Custom Dataset for VQA
class VQADataset(Dataset):
    def __init__(self, questions, image_folder, tokenizer, image_processor, model_config, args, image_aspect_ratio = None):
        self.questions       = questions
        self.image_folder    = image_folder
        self.tokenizer       = tokenizer
        self.image_processor = image_processor
        self.model_config    = model_config
        self.data_args       = args
        self.image_aspect_ratio = getattr(args, "image_aspect_ratio", None)
    
    def __getitem__(self, index):
        line       = self.questions[index]
        image_file = line['image']
        image      = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')

        source_dup = copy.deepcopy(line)

        attempt      = 0
        max_attempts = 10
        while True:
            try:
                image, conversation = visual_prompt_process(source_dup, image, image_size_anchor = self.image_processor.crop_size['height'], data_args = self.data_args)
                break
            except:
                attempt += 1
                if attempt >= max_attempts:
                    print(f"Failed to process image {image_file} after {max_attempts} attempts.")
                    return self.__getitem__(random.randint(0, len(self.questions) - 1))

        question = conversation[0]['value']
        question = question.replace("<image>", "").strip()
        question = DEFAULT_IMAGE_TOKEN + "\n" + question
        gpt_char = conversation[1]['value']

        conv = templates[self.data_args.conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        image_tensors = process_any_resolution_image(
            image          = image,
            processor      = self.image_processor,
            grid_pinpoints = self.model_config.mm_grid_pinpoints,
        )
        image_tensors = image_tensors.to(dtype = torch.float16)
        #print("Image shape:", image_tensors[0].shape)

        input_ids = tokenizer_image_token(
            prompt = prompt,
            tokenizer   = self.tokenizer,
            return_tensors = 'pt',
        )

        return input_ids, image_tensors, gpt_char, prompt
    
    def __len__(self):
        return len(self.questions)

def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, args, batch_size = 1):
    assert batch_size == 1, "Batch size must be 1 for VQA evaluation"
    dataset = VQADataset(
        questions       = questions,
        image_folder    = image_folder,
        tokenizer       = tokenizer,
        image_processor = image_processor,
        model_config    = model_config,
        args            = args,
    )
    return DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers = args.num_workers)

#====================================================================================================================================
# MAIN EVALUATION FUNCTION
#====================================================================================================================================
def eval_model(args):
    # --- 1 --- Load the model -----------------------------------------------------------------------------------------------
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)

    #Load the tokenizer, model, and image processor
    tokenizer, model, image_processor, context_length = load_pretrained_model(
        model_path = args.model_path,
        model_base = args.model_base,
        model_name = model_name,
        load_8bit  = args.load_8bit,
        load_4bit  = args.load_4bit,
        device_map = "auto",
        device     = args.device,
    )

    # --- 2 --- Load the VQA dataset ------------------------------------------------------------------------------------------
    questions = json.load(open(os.path.expanduser(args.question_file)))
    questions = get_chunk(
        questions,
        n = args.num_chunks,
        k = args.chunk_idx
    )
    
    answer_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answer_file), exist_ok = True)
    ans_file = open(answer_file, "w")

    data_loader = create_data_loader(
        questions       = questions,
        image_folder    = args.image_folder,
        tokenizer       = tokenizer,
        image_processor = image_processor,
        model_config    = model.config,
        args            = args,
    )

    correct = 0
    total   = 0
    terminators = [tokenizer.eos_token_id]

    #RUN
    for i, ((input_ids, image_tensor, gpt_char, prompt), line) in tqdm(enumerate(zip(data_loader, questions)), total = len(questions)):
        index = line['id']
        total += 1

        input_ids    = input_ids.to(model.device)
        image_tensor = image_tensor.to(dtype = torch.float16, device = model.device)
        
        stopping_criteria = KeywordsStoppingCriteria(
            keywords  = ["</s>"],
            tokenizer = tokenizer,
            input_ids = input_ids
        )

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids      = input_ids,
                images         = image_tensor,
                max_new_tokens = args.max_new_tokens,
                do_sample      = True if args.temperature > 0 else False,
                temperature    = args.temperature,
                eos_token_id   = terminators,
                use_cache      = True,
                # stopping_criteria = stopping_criteria,
            )
        
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": index,
                                   "question": prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
        
        #Process Output
        outputs = extract_answer(outputs)
        if outputs.lower() == gpt_char[0].lower():
            correct += 1
        #print(f"Accuracy: {correct / (i + 1)}")
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--model-base", type = str, default = None, help = "Optional base model for loading delta weights.")
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--alpha", type=int, default=128)
    parser.add_argument("--image_aspect_ratio", type=str, default=None)
    parser.add_argument("--visual_prompt_style", type=str, default=None)
    parser.add_argument("--load-8bit", action = "store_true", help = "Load the model in 8-bit mode.")
    parser.add_argument("--load-4bit", action = "store_true", help = "Load the model in 4-bit mode.")
    parser.add_argument("--device", type = str, default = "cuda", help = "Device to use (e.g., 'cuda', 'cpu').")
    parser.add_argument("--num_workers", type = int, default = 1)
    
    args = parser.parse_args()
    eval_model(args)