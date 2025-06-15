# =================================================================================================
# File: vis_zephyr/model/vision_encoder/builder.py
# Description: A factory function to construct the vision tower.
# =================================================================================================
import os
from .vision_encoder import CLIPVisionTower

def build_vision_tower(vision_tower_cfg, **kwargs):
    """
    Build the vision tower based on the provided configuration.
    """
    vision_tower_path = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    is_absolute_path_exists = os.path.exists(vision_tower_path)
    is_clip_openai_model    = vision_tower_path.startswith("openai")

    if is_absolute_path_exists or is_clip_openai_model or vision_tower_path.startswith("laion"):
        return CLIPVisionTower(
            vision_tower_path = vision_tower_path,
            args              = vision_tower_cfg,
            **kwargs
        )

    raise ValueError(f'Unknown vision tower path: {vision_tower_path}')