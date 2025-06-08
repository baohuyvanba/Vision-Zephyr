import math
from io import BytesIO
import re
from turtle import back
from typing import List, Tuple, Union

from requests import get
import torch
from PIL import Image
from transformers import StoppingCriteria

from ..constants import IMAGE_TOKEN_INDEX

#IMAGE PROCESSING FUNCTIONS =============================================================================================================
def expand2square(
        pil_image: Image.Image,
        background_color: Tuple[int, int, int],
) -> Image.Image:
    """ Expand an image to a square by adding padding with background color."""
    w, h = pil_image.size
    if w == h:
        #Already square, no padding needed
        return pil_image
    elif w > h:
        #Padding top and bottom
        result = Image.new(pil_image.mode, (w, w), background_color)
        result.paste(pil_image, (0, (w - h) // 2))
        return result
    else:
        #Padding left and right
        result = Image.new(pil_image.mode, (h, h), background_color)
        result.paste(pil_image, ((h - w) // 2, 0))
        return result
    
def process_image(
        images: Union[Image.Image, List[Image.Image]],
        image_processor: object,
        model_config: object,
) -> torch.Tensor:
    """Process image/list of images -> ratio-corrected -> resized -> tensor."""
    if not isinstance(image, Image.Image):
        images = [images]

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
            img = image.resize((target_size, target_size))
            transformed_images.append(img)
    elif aspect_ratio_mode == "square":
        for image in images:
            w, h   = image.size
            new_size = min(image.size)
            left   = (w - new_size) / 2
            top    = (h - new_size) / 2
            right  = (w + new_size) / 2
            bottom = (h + new_size) / 2
            img = image.crop((left, top, right, bottom))
            transformed_images.append(img)
    else:
        transformed_images = images

    processed_images = image_processor(images = transformed_images, return_tensors='pt')['pixel_values']
    
    return processed_images

#PROCESS INPUT PROMPT (WITH IMAGE TOKEN) ================================================================================================
def tokenize_image_token(
        prompt: str,
        tokenizer: object,
        image_token_index: int = IMAGE_TOKEN_INDEX,
        return_tensor: str = None,
) -> Union[List[int], torch.LongTensor]:
    """Input Prompt -> Split by <image> placeholder -> Tokenize text parts -> Add specific image token"""

    #Split input prompt by <image> placeholder and Tokenize each
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    #Add image token to each chunk
    def insert_separator(X, sep):
        return [element for sublist in zip(X, [sep]*len(X)) for element in sublist][:-1]
    
    input_ids = []
    offset = 0

    #BOS token handling
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1 #Remove BOS token except first chunk
        input_ids.append(prompt_chunks[0][1:]) #Add first chunk without BOS token

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensor is not None:
        if return_tensor == 'pt':
            return torch.tensor(input_ids, dtype = torch.long)
        raise ValueError(f"Unknown return_tensor type: {return_tensor}")
    return input_ids
    
def get_model_name_from_path(model_path: str):
    """Extract model name from a given path."""
    # model_path  = model_path.strip('/')
    model_paths = model_path.split('/')
    if model_paths[-1].startswith('checkpoint-'):
        return f"{model_paths[-2]}_{model_paths[-1]}"
    else:
        return model_paths[-1]
    
#STOPPING CRITERIA =======================================================================================================================
class KeywordStoppingCriteria(StoppingCriteria):
    """Stop generation when a specific keyword is generated."""
    
    def __init__(
            self,
            keywords: List[str],
            tokenizer: object,
            input_ids: torch.LongTensor
    ):
        self.keywords = keywords
        self.tokenizer = tokenizer
        
        #Get keyword's ids
        self.keyword_ids = []
        for keyword in keywords:
            current_keyword_ids = tokenizer(keyword).input_ids
            #Remove BOS token if present
            if current_keyword_ids[0] == tokenizer.bos_token_id:
                current_keyword_ids = current_keyword_ids[1:]
            self.keyword_ids.append(torch.tensor(current_keyword_ids))
        
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

#Any-Resulation Processing Functions ====================================================================================================
###