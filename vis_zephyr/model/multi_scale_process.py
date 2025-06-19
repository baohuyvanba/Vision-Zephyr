# =================================================================================================
# File: vis_zephyr/model/multi_scale_process.py
# Description: Any-Resolation Processing Functions
# =================================================================================================
import ast
from calendar import c
from typing import List, Tuple
import torch
from PIL import Image

def _robust_literal_eval(value_str):
    """
    Recursively evaluates a string literal until it's no longer a string.
    Handles cases like "'[[1, 2]]'" being passed from command line.
    """
    if not isinstance(value_str, str):
        return value_str
    
    res = value_str
    while isinstance(res, str):
        try:
            res = ast.literal_eval(res)
        except (ValueError, SyntaxError):
            return res
    return res

#Select the best fit resolution from a list of possible resolutions --------------------------------------------------------------------
def select_best_fit_resolution(
    original_resolution : Tuple[int, int],
    possible_resolutions: List[Tuple[int, int]],
) -> Tuple[int, int]:
    """
    Select the best fit resolution from a list of possible resolutions.
    The best fit is the one that is closest to the original resolution.
    """
    ori_width, ori_height = original_resolution
    
    best_fit_res      = None
    #Maximum effective resolution achieved by fitting original image -> possible resolution
    # (without being cropped or scaled down excessively)
    max_effective_res = 0              #to Maximize
    #Minimum wasted resolution (difference between original and possible resolution)
    min_wasted_res    = float('inf')   #to Minimize

    for w, h in possible_resolutions:
        #Calulate the Scaling factor to fit: get min scaling -> "fit-within"
        scale = min(w / ori_width, h / ori_height)
        #Downscaled dim after scaling
        downscaled_w, downscaled_h = int(ori_width * scale), int(ori_height * scale)
        #Calculate effective resolution: the area of the downscaled image
        effective_res = min(downscaled_w*downscaled_h, ori_width*ori_height)

        wasted_res    = (w*h) - effective_res
        
        #Choose the resolution with the: Maximum effective resolution -> Minimum wasted resolution
        if effective_res > max_effective_res or (
            effective_res == max_effective_res and wasted_res < min_wasted_res
        ):
            max_effective_res = effective_res
            min_wasted_res    = wasted_res
            best_fit_res      = (w, h)
        
    return best_fit_res

#Resize and pad an image to a target resolution ----------------------------------------------------------------------------------------
def resize_pad_image(
        image: Image.Image,
        target_res: Tuple[int, int],
) -> Image.Image:
    """
    Resize and Pad image -> target resolution.
    """
    original_w, original_h = image.size
    target_W, target_H     = target_res
    #Scaling factors
    scale_w, scale_h = target_W / original_w, target_H / original_h
    scale_factor = min(scale_w, scale_h)

    new_w, new_h = int(original_w * scale_factor), int(original_h * scale_factor)
    resized_image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    new_image = Image.new('RGB', (target_W, target_H), (0, 0, 0))  # Black background
    paste_w, paste_h = (target_W - new_w) // 2, (target_H - new_h) // 2
    new_image.paste(resized_image, (paste_w, paste_h))

    return new_image

#Divide an image into patches of specified size ---------------------------------------------------------------------------------------
def divide_to_patches(
        image: Image.Image,
        patch_size: Tuple[int, int],
) -> List[Image.Image]:
    """
    Divide an image into patches of specified size.
    """
    patches = []
    w, h    = image.size
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)
    return patches

#Calculate the grid shape based on the image size, grid pinpoints, and patch size -----------------------------------------------------
def calculate_grid_shape(
        image_size    : Tuple[int, int],
        grid_pinpoints: str, #List of potential grid resolutions
        patch_size    : int, #Images -> Patchs with patch_size
) -> Tuple[int, int]:
    """
    Calculate the grid shape based on the image size, grid pinpoint, and patch size.
    -> Numbers of rows and columns in the grid.
    """
    possible_res = _robust_literal_eval(grid_pinpoints)
    if not isinstance(possible_res, list):
        raise ValueError(f"grid_pinpoints did not evaluate to a list: {grid_pinpoints}")

    w, h = select_best_fit_resolution(image_size, possible_res)
    
    return (w // patch_size, h // patch_size)

#Process an image with any resolution, resizing and padding it -> return tensor of patches --------------------------------------------
def process_any_resolution_image(
    image         : Image.Image,
    processor     : object,
    grid_pinpoints: str,
) -> torch.Tensor:
    """
    Process an image with any resolution.
    """
    if isinstance(grid_pinpoints, list):
        possible_resolution = grid_pinpoints
    else:
        possible_resolution = _robust_literal_eval(grid_pinpoints)

    #Get the best fit resolution
    best_fit_res = select_best_fit_resolution(
        original_resolution  = image.size,
        possible_resolutions = possible_resolution
    )
    #Resize image with padding to best fit resolution
    image_padded = resize_pad_image(
        image      = image,
        target_res = best_fit_res
    )
    #Divide padded image into patches
    patches_from_image = divide_to_patches(
        image      = image_padded,
        patch_size = processor.crop_size['height']
    )
    #Resize original image (base image) -> to the shortest edge of the crop size
    resized_original_image = image.resize(
        (processor.crop_size['height'], processor.crop_size['height']),
        Image.Resampling.LANCZOS
    )
    #List of Patches = [Resized Image] and list of Image Patches -> Preprocess them
    patches = [resized_original_image] + patches_from_image
    preprocessed_patches = [
        processor.preprocess(patch, return_tensors='pt')['pixel_values']
        for patch in patches
    ]

    return torch.stack(preprocessed_patches, dim= 0)

#======================================================================================================================================
# UNPAD IMAGE: remove padding from padded/resized image
#======================================================================================================================================
def unpad_image(
    image_tensor,
    original_size,
):
    original_w, original_h = original_size
    current_w , current_h  = image_tensor.shape[1:]

    original_aspect_ratio  = original_w / original_h
    current_aspect_ratio   = current_w / current_h

    if original_aspect_ratio > current_aspect_ratio:
        # Original is wider than current
        factor   = current_w / original_w
        new_h    = int(original_h * factor)
        padding  = (current_h - new_h) // 2
        unpadded = image_tensor[:, padding:current_h - padding, :]
    else:
        # Original is taller than current
        factor   = current_h / original_h
        new_w    = int(original_w * factor)
        padding  = (current_w - new_w) // 2
        unpadded = image_tensor[:, :, padding:current_w - padding]
    
    return unpadded