import argparse
import torch
import requests

from PIL import Image
from io import BytesIO
from transformers import TextStreamer

from vis_zephyr.constants import *
from vis_zephyr.conservation import templates, SeparatorStyle
from vis_zephyr.model.builder import load_pretrained_model
from vis_zephyr.utils import disable_torch_init
from vis_zephyr.model.mm_utils import *

def load_image(image_file: str) -> Image.Image:
    """Load image from a file path or URL"""
    if image_file.startswith('http://') or image_file.startswith('https://'):
        #Get image from URL
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        #Get image from local file
        image = Image.open(image_file).convert('RGB')
    return image

def main(args):
    """Command line interface"""
    #Initialize the model
    disable_torch_init()
    model_name = get_model_name_from_path(args.model_path)

    # Load the tokenizer, model, and image processor
    tokenizer, model, image_processor, context_length = load_pretrained_model(
        model_path = args.model_path,
        model_base = args.model_base,
        model_name = model_name,
        load_8bit  = args.load_8bit,
        load_4bit  = args.load_4bit,
        device_map = "auto",
        device     = "cuda",
    )

    # Set the conversation template
    if args.conv_mode is None:
        args.conv_mode = "zephyr_v1"

    #Conservation
    conversation  = templates[args.conv_mode].copy()
    roles         = conversation.roles

    #Load and process the image
    image = load_image(args.image_file)
    image_tensor = process_image(images = image,
                                 image_processor = image_processor,
                                 model_config = model.config)
    image_tensor = image_tensor.to(model.device, dtype=torch.float16)

    #INTERACTIVE CHAT LOOP
    while True:
        try:
            input = input(f"{roles[0]}: ")
        except EOFError:
            input = ""
        if not input or input.lower() in ["exit", "quit"]:
            print("Exiting chat.")
            break

        print(f"{roles[1]}: ", end = "")

        if image is not None:
            #First message must contain an image
            if model.config.mm_use_im_start_end:
                input = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + input
            else:
                input = DEFAULT_IMAGE_TOKEN + "\n" + input

            conversation.append_message(conversation.roles[0], input)
            image = None
        else:
            #Text-only message
            conversation.append_message(conversation.roles[0], input)

        conversation.append_message(conversation.roles[1], None)
        prompt = conversation.get_prompt()

        #Tokenize the prompt (with image) and add the image token
        input_ids = tokenize_image_token(
            prompt = prompt,
            tokenizer = tokenizer,
            image_token_index = IMAGE_TOKEN_INDEX,
            return_tensor = 'pt',
        ).unsqueeze(0).to(model.device)

        stop_str = conversation.separator_01 if conversation.separator_style != SeparatorStyle.TWO else conversation.separator_02
        keywords = [stop_str]
        stopping_criteria = KeywordStoppingCriteria(
            keywords  = keywords,
            tokenizer = tokenizer,
            input_ids = input_ids
        )
        streamer = TextStreamer(
            tokenizer = tokenizer,
            skip_prompt = True,
            skip_special_tokens = True
        )

        #Generate the response
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids = input_ids,
                images = image_tensor,
                do_sample = True if args.temperature > 0 else False,
                temperature = args.temperature,
                max_new_tokens = args.max_new_tokens,
                streamer = streamer,
                use_cache = True,
                stopping_criteria = [stopping_criteria]
            )

        #Decode the output
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conversation.messages[-1][1] = outputs

        if args.debug:
            print("\n\n[Debug] Prompt:")
            print("\n[DEBUG] Final Output:\n", outputs, "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model checkpoint or HF repo.")
    parser.add_argument("--model-base", type=str, default=None, help="Optional base model for loading delta weights.")
    parser.add_argument("--image-file", type=str, required=True, help="Path to the input image file.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (e.g., 'cuda', 'cpu').")
    parser.add_argument("--conv-mode", type=str, default=None, help="Conversation mode template.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum number of new tokens to generate.")
    parser.add_argument("--load-8bit", action="store_true", help="Load the model in 8-bit mode.")
    parser.add_argument("--load-4bit", action="store_true", help="Load the model in 4-bit mode.")
    parser.add_argument("--debug", action="store_true", help="Print debug information.")
    args = parser.parse_args()
    main(args)