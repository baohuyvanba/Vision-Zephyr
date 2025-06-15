# =================================================================================================
# File: vis_zephyr/utils.py
# Description: General utility functions for the project.
# =================================================================================================

def disable_torch_init():
    """Disable torch initialization -> speed up model loading"""
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)