# =================================================================================================
# File: vis_zephyr/serve/cli.py
# Description: Command-line interface for running interactive inference with the Vision-Zephyr model.
# =================================================================================================
import argparse
from numpy import isin
import torch
import requests

from PIL import Image
from io import BytesIO
from transformers import TextStreamer

from vis_zephyr.constants import *
from vis_zephyr.conversation import templates, SeparatorStyle
from vis_zephyr.model.builder import load_pretrained_model
from vis_zephyr.model.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from vis_zephyr.utils import disable_torch_init

#=========================================================================================================================
# Load Image from path of URL
#=========================================================================================================================
def load_image(image_file: str) -> Image.Image:
    """Load image from a file path or URL."""
    #Get image from URL
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    #Get image from local file
    else:
        image = Image.open(image_file).convert('RGB')
    return image

#=========================================================================================================================
# Main function to run the CLI for interactive inference
#=========================================================================================================================
def main(args):
    """Command line interface."""
    # --- 1 --- Initialize the model
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
        device     = args.device, #"cuda"
    )

    # --- 2 --- Conversation Setup
    conv_mode = "zephyr_v1"
    if args.conv_mode is not None and args.conv_mode != conv_mode:
        print(f"[WARNING] The auto-inferred conversation mode is {conv_mode}, but --conv-mode is set to {args.conv_mode}. Using {conv_mode}.")
        args.conv_mode = conv_mode
    
    conversation  = templates[conv_mode].copy()
    roles         = conversation.roles

    # --- 3 --- Image Processing
    image = load_image(args.image_file)
    images_size = image.size

    #Pre-process the image based on the model's configuration
    if model.config.image_aspect_ratio == 'anyres':
        from vis_zephyr.model.multi_scale_process import process_any_resolution_image
        image_tensor = process_any_resolution_image(
            image          = image,
            processor      = image_processor,
            grid_pinpoints = model.config.mm_grid_pinpoints,
        )
    else:
        image_tensor = process_images(
            images          = image,
            image_processor = image_processor,
            model_config    = model.config
        )
        if isinstance(image_tensor, list):
            image_tensor = image_tensor[0]
    
    #Move the image tensor to the correct device and dtype
    if isinstance(image_tensor, list):
        image_tensor = [
            img.to(model.device, dtype = torch.float16)
            for img in image_tensor
        ]
    else:
        image_tensor = image_tensor.unsqueeze(0).to(model.device, dtype = torch.float16)

    # --- 4 --- INTERACTIVE CHAT LOOP
    while True:
        try:
            user_input = input(f"{roles[0]}: ")
        except EOFError:
            user_input = ""
        #Exit condition
        if not user_input or user_input.lower() in ["exit", "quit"]:
            print("Exiting chat...")
            break
        
        #Print the assistant's role
        print(f"{roles[1]}: ", end = "")

        #USER INPUT PROCESSING
        if image is not None:
            #First message must contain an image (prepend the image token to the user input)
            #PROBLEM
            if model.config.mm_use_im_start_end:
                user_input = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + user_input
            else:
                user_input = DEFAULT_IMAGE_TOKEN + "\n" + user_input
            conversation.append_message(conversation.roles[0], user_input)
            #Only sent the image once
            image = None
        else:
            #Text-only message
            conversation.append_message(conversation.roles[0], user_input)

        #ASSISTANT RESPONSE placeholder
        conversation.append_message(conversation.roles[1], None)
        prompt = conversation.get_prompt() #Get: "<|system|>...</s><|user|>...</s><|assistant|>...</s>..." -> Model

        #Tokenize the prompt (with image) and add the image token
        input_ids = tokenizer_image_token(
            prompt            = prompt,
            tokenizer         = tokenizer,
            image_token_index = IMAGE_TOKEN_INDEX,
            return_tensors    = 'pt',
        ).unsqueeze(0).to(model.device)

        #Stopping criteria setup
        stop_str = conversation.separator_01 if conversation.separator_style == SeparatorStyle.ZEPHYR else conversation.separator_02
        print("=== STOP STRING IS === ", stop_str, "")
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords  = keywords,
            tokenizer = tokenizer,
            input_ids = input_ids
        )
        #Real-time text streaming setup
        streamer = TextStreamer(
            tokenizer   = tokenizer,
            skip_prompt = True,
            skip_special_tokens = True
        )

        #Generate the response (INFERENCE MODE)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids      = input_ids,
                images         = image_tensor,
                do_sample      = True if args.temperature > 0 else False,
                temperature    = args.temperature,
                max_new_tokens = args.max_new_tokens,
                streamer       = streamer,
                use_cache      = True,
                stopping_criteria = [stopping_criteria]
            )

        #Decode the output
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)].strip()
        
        #Update the conversation with the generated response
        conversation.messages[-1][1] = outputs

        if args.debug:
            print("\n\n[Debug] Prompt:")
            print("\n[DEBUG] Final Output:\n", outputs, "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type = str, required = True, help = "Path to the model checkpoint or HF repo.")
    parser.add_argument("--model-base", type = str, default = None, help = "Optional base model for loading delta weights.")
    parser.add_argument("--image-file", type = str, required = True, help = "Path to the input image file.")
    parser.add_argument("--device", type = str, default = "cuda", help = "Device to use (e.g., 'cuda', 'cpu').")
    parser.add_argument("--conv-mode", type = str, default = None, help = "Conversation mode template.")
    parser.add_argument("--temperature", type = float, default = 0.2, help = "Sampling temperature.")
    parser.add_argument("--max-new-tokens", type = int, default = 512, help = "Maximum number of new tokens to generate.")
    parser.add_argument("--load-8bit", action = "store_true", help = "Load the model in 8-bit mode.")
    parser.add_argument("--load-4bit", action = "store_true", help = "Load the model in 4-bit mode.")
    parser.add_argument("--debug", action = "store_true", help = "Print debug information.")
    
    args = parser.parse_args()
    main(args)