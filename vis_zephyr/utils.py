import logging
import sys

def disable_torch_init():
    """Disable torch initialization to prevent unnecessary warnings."""
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)