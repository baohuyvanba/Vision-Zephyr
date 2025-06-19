# =================================================================================================
# File: vis_zephyr/model/builder.py
# Description: Handles the loading of pre-trained Vision-Zephyr models (including LLM Backbone, Projector, Vision Encoder).
# =================================================================================================
from logging import config
import os
import torch
from transformers import AutoTokenizer, AutoConfig, BitsAndBytesConfig

from .language_model import VisZephyrForCausalLM
from vis_zephyr.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

#---------------------------------------------------------------------------------------------------------------------------------------
# Load a pretrained model from a given path or Hugging Face model name.
#----------------------------------------------------------------------------------------------------------------------------------------
def load_pretrained_model(
        model_path, # Path to the model checkpoints
        model_base, # Path to the LLM backbone
        model_name, # Name of the model, used for identifying the type of model and inference architect
        load_8bit: bool = False,
        load_4bit: bool = False,
        device_map = {"": "cuda:0"},
        device     = "cuda",
        #model_args = None,
        **kwargs
):
    """
    Load a pretrained model from a given path or Hugging Face model name.
    """
    kwargs = {} #"device_map": device_map, **kwargs}
    if device != "cuda":
        kwargs['device_map'] = None #{"": device}
    
    #Quantization Settings
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
    #>>> Load LoRA finetuned Model
    if 'lora' in model_name.lower() and model_base is not None:
        print("=== Loading Zephyr LLM Backbone with LoRA from base path ===")
        from peft import PeftModel

        #Loading the configuration
        lora_cfg_pretrained = AutoConfig.from_pretrained(model_path)
        #Loading the tokenizer from base LLM backbone
        tokenizer           = AutoTokenizer.from_pretrained(model_base, use_fast = False)

        #Loading the model with LoRA specific configuration
        model = VisZephyrForCausalLM.from_pretrained(
            model_base,
            low_cpu_mem_usage = True,
            config            = lora_cfg_pretrained, #LoRA specific configuration
            **kwargs
        )

        #Resize token embeddings
        token_num, token_dim = model.lm_head.out_features, model.lm_head.in_features
        if model.lm_head.weight.shape[0] != token_num:
            model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, token_dim, device = model.device, dtype = model.dtype))
            model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, token_dim, device = model.device, dtype = model.dtype))
        
        #Load non-LoRA weights: MultiModal Projector
        print("=== Loading MultiModal Projector ===")
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location = 'cpu')
        
        #Clean up the keys (cause by standard DeepSpeed/FSDP) to match the model's state_dict
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.starswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[12:] if k.startswith('model.model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict = False)

        #Load LoRA Adapters
        print("=== Loading LoRA Adapters ===")
        model = PeftModel.from_pretrained(
            model,
            model_path,
        )

        #Merge LoRA weights into the model
        print("=== Merging LoRA weights into the model ===")
        model = model.merge_and_unload()

    #>>> Load Model without LoRA
    elif model_base is not None:
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
    
    #>>> Load fully consolidated model
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model     = VisZephyrForCausalLM.from_pretrained(
            model_path,
            low_cpu_mem_usage = True,
            **kwargs
        )
    
    #LOAD Vision Encoder ==========================================================================================================
    model.get_model().initialize_vision_modules(model_args = cfg_pretrained)
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        print("=== Initializing Vision Encoder ===")
        vision_tower.load_model()
    vision_tower.to(device = device, dtype = torch.float16)
    image_processor = vision_tower.image_processor

    #Add special tokens for image processing
    mm_use_im_start_end   = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    
    #Context Length
    if hasattr(model.config, "max_squence_length"):
        context_length = model.config.max_squence_length
    else:
        context_length = 2048

    return tokenizer, model, image_processor, context_length