# =================================================================================================
# File: vis_zephyr/model/vip_processor/processor.py
# Description: Handles Visual Prompt processing for Vision-Zephyr models.
# =================================================================================================

def visual_prompt_process(
    source,
    image,
    image_size_anchor,
    data_args,
):
    conversation = source['conversation']
    return image, conversation
