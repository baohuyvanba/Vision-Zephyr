# =================================================================================================
# File: vis_zephyr/model/mm_utils.py
# Description: Utility functions for multimodal processing;
# Including: Image and Token manipulation; Image's resolution handling.
# =================================================================================================

import ast
from typing import List, Tuple, Union

import torch
from PIL import Image
from transformers import StoppingCriteria

from ..constants import IMAGE_TOKEN_INDEX

#=====================================================================================================================================
# IMAGE PROCESSING FUNCTIONS 
#=====================================================================================================================================

def expand2square(
        pil_image: Image.Image,
        background_color: Tuple[int, int, int],
) -> Image.Image:
    """
    Expand an image (Padding with color) -> Square image.
    """
    w, h = pil_image.size
    if w == h:
        #Already square
        return pil_image
    elif w > h:
        #Padding top and bottom (modify height)
        result = Image.new(pil_image.mode, (w, w), background_color)
        result.paste(pil_image, (0, (w - h) // 2))
        return result
    else:
        #Padding left and right (modify width)
        result = Image.new(pil_image.mode, (h, h), background_color)
        result.paste(pil_image, ((h - w) // 2, 0))
        return result
    
def process_images(
    images: Union[Image.Image, List[Image.Image]],
    image_processor: object,
    model_config: object,
) -> torch.Tensor:
    """
    Processes an image/list of images: Applying aspect ratio correction -> tensor.
    """
    # Normalize the input to always be a list.
    is_single_image = isinstance(images, Image.Image)
    if is_single_image:
        images = [images]

    # Aspect ratio correction based on the specified mode.
    aspect_ratio_mode = getattr(model_config, 'aspect_ratio_mode', 'square')
    
    transformed_images = []
    if aspect_ratio_mode == "pad":
        background_color = tuple(int(x * 255) for x in image_processor.image_mean)
        for image in images:
            img = expand2square(image, background_color)
            transformed_images.append(img)
    elif aspect_ratio_mode == "resize":
        target_size = image_processor.crop_size['height']
        for image in images:
            img = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
            transformed_images.append(img)
    elif aspect_ratio_mode == "square":
        for image in images:
            width, height = image.size
            new_size = min(width, height)
            left = int((width - new_size) / 2)
            top = int((height - new_size) / 2)
            right = left + new_size
            bottom = top + new_size
            img = image.crop((left, top, right, bottom))
            transformed_images.append(img)
    else:
        transformed_images = images

    #Convert the list of PIL images to a batched tensor:normalization and tensor conversion.
    pixel_values = image_processor(
        images = transformed_images, 
        return_tensors = 'pt'
    )['pixel_values']
    
    if is_single_image:
        return pixel_values[0]
    else:
        return pixel_values
#======================================================================================================================================
# PROCESS INPUT PROMPT (WITH IMAGE TOKEN)
#======================================================================================================================================
def tokenizer_image_token(
        prompt: str,
        tokenizer: object,
        image_token_index: int = IMAGE_TOKEN_INDEX,
        return_tensor: str = None,
):
    """Input Prompt -> Split into chunks by <image> placeholder -> Tokenize text parts -> Add specific image token (-200)"""

    #Split input prompt by <image> placeholder and Tokenize each
    prompt_chunks = [
        tokenizer(chunk).input_ids
        for chunk in prompt.split('<image>')
    ]

    def insert_separator(X, sep):
        """Inserts a separator between each element of X"""
        return [element for sublist in zip(X, [sep]*len(X)) for element in sublist][:-1]
    
    input_ids = []
    offset = 0
    #Get BOS token index in the first chunk only, if it exists
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    #Add all chunks to input_ids, inserting the image token index at the appropriate places
    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        # [Image Token Index] need to times with (offset + 1) (x2) beacause one of them will be cut with [offset:] ([1:] in [image_toke_index, image_token_index]) 
        input_ids.extend(x[offset:])

    if return_tensor is not None:
        if return_tensor == 'pt':
            return torch.tensor(input_ids, dtype = torch.long)
        raise ValueError(f"Unknown return_tensor type: {return_tensor}")
    return input_ids

#MODEL NAME EXTRACTION ===================================================================================================================
def get_model_name_from_path(model_path: str):
    """Extract model name from a given path."""
    model_path  = model_path.strip('/')
    model_paths = model_path.split('/')
    if model_paths[-1].startswith('checkpoint-'):
        return f"{model_paths[-2]}_{model_paths[-1]}"
    else:
        return model_paths[-1]

#======================================================================================================================================
# GENERATION STOPPING CRITERIA
#======================================================================================================================================
class KeywordsStoppingCriteria(StoppingCriteria):
    """
    Stop generation when a specific keyword is generated.
    """
    
    def __init__(
            self,
            keywords: List[str],
            tokenizer: object,
            input_ids: torch.LongTensor
    ):
        self.keywords    = keywords
        self.keyword_ids = []
        self.max_length  = 0
        for keyword in keywords:
            current_keyword_ids = tokenizer(keyword).input_ids
            
            if len(current_keyword_ids[0]) > 1 and current_keyword_ids[0] == tokenizer.bos_token_id:
                #Remove the first token if it is a BOS token
                current_keyword_ids = current_keyword_ids[1:]
            if len(current_keyword_ids) > self.max_length:
                #Update max_length if current keyword is longer
                self.max_length = len(current_keyword_ids)
            self.keyword_ids.append(torch.tensor(current_keyword_ids))
        
        self.tokenizer = tokenizer
        self.start_length = input_ids.shape[1]

    def __call__(
            self,
            output_ids: torch.LongTensor,
            scores: torch.FloatTensor,
            **kwargs
    ) -> bool:
        """Check if any of the keywords are in the generated output."""
        for i in range(output_ids.shape[0]):
            generated_ids = output_ids[i, self.start_length:]
            for keyword_id in self.keyword_ids:
                if len(generated_ids) >= len(keyword_id):
                    if (generated_ids[-len(keyword_id):].cpu() == keyword_id).all():
                        return True
        return False
    
    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
    
    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)