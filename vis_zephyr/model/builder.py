import os
import warnings
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig

from vis_zephyr.model import *

def load_pretrained_model(
        model_path,
        model_base,
        model_name,
        load_8bit  = False,
        load_4bit  = False,
        device_map = "auto",
        device     = "cuda",
        **kwargs
):
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

    if "zephyr" in model_name.lower():
        #Load Vis-Zephyr model
        if model_base is not None:
            print("=== Loading Zephyr LLM Backbone from base ===")
            tokenizer      = AutoTokenizer.from_pretrained(model_base, use_fast = False)
            cfg_pretrained = AutoConfig.from_pretrained(model_path)
            
            #The model class is dynamically registered, so AutoModelForCausalLM can find it.
            model = VisZephyrForCausalLM.from_pretrained(model_base,
                                                                low_cpu_mem_usage = True,
                                                                config = cfg_pretrained,
                                                                **kwargs)
            
            #Load MultiModal Projector
            print("=== Loading MultiModal Projector ===")
            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict = False)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model     = VisZephyrForCausalLM.from_pretrained(
                model_path,
                low_cpu_mem_usage = True,
                **kwargs
            )
    else:
        raise ValueError(f"Unsupported model name: {model_name}. ")
    
    #Initialize Vision Components
    if hasattr(model.config, "mm_vision_tower"):
        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded():
            print("=== Initializing Vision Tower ===")
            vision_tower.load_model()
        
        vision_tower.to(device = device, dtype = torch.float16)
        image_processor = vision_tower.image_processor
    else:
        #Pure LLM only, WARNING
        image_processor = None
        warnings.warn("No vision tower found in the model. This is a pure LLM model without vision components.")
    
    if hasattr(model.config, "max_squence_length"):
        context_length = model.config.max_squence_length
    else:
        context_length = 2048

    return tokenizer, model, image_processor, context_length
