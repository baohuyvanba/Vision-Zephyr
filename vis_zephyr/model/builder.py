# =================================================================================================
# File: vis_zephyr/model/builder.py
# Description: Handles the loading of pre-trained Vision-Zephyr models (including LLM Backbone, Projector, Vision Encoder).
# =================================================================================================
import os
import torch
from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig

from .language_model import VisZephyrForCausalLM

#=======================================================================================================================================
# Load a pretrained model from a given path or Hugging Face model name.
#=======================================================================================================================================
def load_pretrained_model(
        model_path, # Path to the model checkpoints
        model_base, # Path to the LLM backbone
        model_name, # Name of the model, used for identifying the type of model and inference architect
        load_8bit: bool = False,
        load_4bit: bool = False,
        device_map = "auto",
        device     = "cuda",
        model_args = None,
        **kwargs
):
    """
    Load a pretrained model from a given path or Hugging Face model name.
    """
    kwargs = {"device_map": device_map, **kwargs}
    if device != "cuda":
        kwargs['device_map'] = {"": device}
    
    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16
    
    if "zephyr" not in model_name.lower():
        raise ValueError(f"Unsupported model name: {model_name}. Only Zephyr models are supported at the moment.")

    #LOAD LLM Backbone + Projector model ============================================================================================
    #>>> Load pretrained-LLM + new/pretrained Projector
    if model_base is not None:
        #Load Vis-Zephyr LLM backbone model -----------------------------------------------------------------------------------------
        print("=== Loading Zephyr LLM Backbone from base path ===")
        tokenizer      = AutoTokenizer.from_pretrained(model_base, use_fast = False)
        cfg_pretrained = AutoConfig.from_pretrained(model_path)
        
        #The model class is dynamically registered -> AutoModelForCausalLM can find it.
        model = VisZephyrForCausalLM.from_pretrained(
            model_base,
            low_cpu_mem_usage = True,
            config = cfg_pretrained,
            **kwargs
        )
        
        #Load MultiModal Projector -------------------------------------------------------------------------------------------------
        print("=== Loading MultiModal Projector ===")
        mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
        mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
        model.load_state_dict(mm_projector_weights, strict = False)
    else:
    #>>> Load fully consolidated model
        #Load Model from path ------------------------------------------------------------------------------------------------------
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model     = VisZephyrForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage = True,
            **kwargs
        )
    
    #LOAD Vision Encoder ==========================================================================================================
    model.get_model().initialize_vision_modules(model_args = model_args)
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded():
        print("=== Initializing Vision Encoder ===")
        vision_tower.load_model()
    vision_tower.to(device = device, dtype = torch.float16)
    image_processor = vision_tower.image_processor
    
    #Context Length
    if hasattr(model.config, "max_squence_length"):
        context_length = model.config.max_squence_length
    else:
        context_length = 2048

    return tokenizer, model, image_processor, context_length